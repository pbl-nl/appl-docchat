import json
import re
from os import makedirs, path

from flask import flash, url_for, redirect, send_file, jsonify
from flask_login import current_user
from flask_wtf import FlaskForm
from wtforms import HiddenField, SelectField, StringField, SubmitField
from wtforms.validators import Length, ValidationError
from sqlalchemy import delete

import flask_app.models.permissions as permission
from flask_app.models import db, Chat, ChatQuestion, DocSet
from flask_app.helpers import render_chat_template

from query.querier import Querier
from langchain.schema import AIMessage, HumanMessage


class ChatForm(FlaskForm):

    # Field definitions
    id = HiddenField('ID')
    name = StringField('Name', [Length(min=3, max=64)])
    docset_id = SelectField('Document set', coerce=int)
    submit = SubmitField('Save')

    # Custom validation    ( See: https://wtforms.readthedocs.io/en/stable/validators/ )
    def validate_name(form, field):
        if not re.search(r'^[a-zA-Z0-9-_ ]+$', field.data):
            raise ValidationError('Invalid name; Only letters, digits, spaces, - _ characters allowed.')
        same_chat = Chat.query.filter(Chat.name == field.data.strip(), Chat.id != form.chat_id_for_validation).all()
        if len(same_chat) >= 1:
            raise ValidationError('This name already exists.')
        
    def validate_docset_id(form, field):
        if not field.data >= 1:
            raise ValidationError('Invalid docset.')


    # Handle the request (from routes.py) for this form
    def handle_request(self, method, id, extra_parms=None):

        # Show the form
        if method == 'GET':
            self.docset_id.choices = permission.my_docsets()
            if id > 0:
                # Get record from database and set the form values (if id == 0 the defaults are used)
                obj = Chat.query.get(id)
                obj.fields_to_form(self)
                self.chatquestions = ChatQuestion.query.filter(ChatQuestion.chat_id == id).all()
                docset = DocSet.query.filter(DocSet.id == obj.docset_id).first()
            else:
                docset = None
            # Show the form
            return render_chat_template('chat.html', form=self, docset=docset)
        
        # Save the form
        if method == 'POST':
            self.chat_id_for_validation = id
            self.docset_id.choices = [(docset.id, docset.name) for docset in DocSet.query.all()]
            if self.validate():
                if id >= 1:
                    # The table needs to be updated with the new values
                    obj = Chat.query.get(id)
                    obj.fields_from_form(self)
                    db.session.commit()

                else:
                    # A new record must be inserted in tyhe table
                    obj = Chat()
                    obj.fields_from_form(self)
                    db.session.add(obj)
                    db.session.commit()
                    id = obj.id
                
                # Reload the page (in case of an update), or goto the page (in case of an insert)
                return redirect(url_for('chat', id=id))
            
            # Validation failed: Show the form with the errors
            if id > 0:
                # Get record from database and set the form values (if id == 0 the defaults are used)
                obj = Chat.query.get(id)
                docset = DocSet.query.filter(DocSet.id == obj.docset_id).first()
            else:
                docset = None
            return render_chat_template('chat.html', form=self, docset=docset)

        # Delete the chat
        if method == 'DELETE':
            Chat.query.filter(Chat.id == id).delete()
            db.session.execute(ChatQuestion.__table__.delete().where(ChatQuestion.chat_id == id))
            db.session.commit()
            flash('The chat is deleted.', 'info')
            return redirect(url_for('chats'))

        # Specific for chat; Clear the chat history
        if method == 'CLEAR':
            db.session.execute(ChatQuestion.__table__.delete().where(ChatQuestion.chat_id == id))
            db.session.commit()
            flash('The chat history is deleted.', 'info')
            return redirect(url_for('chat', id=id))

        # Specific for chat; Download
        if method == 'DOWNLOAD':
            filename = path.realpath(path.join(path.dirname(__file__), '../..', 'users', 'user-' + str(current_user.id)))
            makedirs(filename, exist_ok=True)
            filename = path.join(filename, 'download.csv')
            file = open(filename, 'w')

            def clean(s):
                return s.replace('"', '\'')
            chatquestions = ChatQuestion.query.filter(ChatQuestion.chat_id == id).all()
            for chatquestion in chatquestions:
                file.write(str(chatquestion.id) + ',"' + str(chatquestion.date) + '","' + clean(chatquestion.question) + '","' + str(chatquestion.date_answer) + '","' + clean(chatquestion.answer) + '","' + clean(chatquestion.source) + '"\n')
            file.close()
            return send_file(filename, download_name='output.csv', as_attachment=True)
        
        # Specific for chat; Ask a question
        if method == 'QUESTION':
            chat = Chat.query.filter(Chat.id == id).first()
            docset = DocSet.query.filter(DocSet.id == chat.docset_id).first()
            collection_name = 'docset_' + str(chat.docset_id)
            question = extra_parms['question']
            past_questions = ChatQuestion.query.filter(ChatQuestion.chat_id == id).all()

            # Create instance of Querier once
            querier = Querier(
                llm_type=docset.llm_type,
                llm_model_type=docset.llm_modeltype,
                embeddings_provider=docset.embeddings_provider,
                embeddings_model=docset.embeddings_model,
                vecdb_type=docset.vecdb_type,
                chain_name=docset.chain,
                chain_type=docset.chain_type,
                chain_verbosity=docset.chain_verbosity,
                search_type=docset.search_type,
                chunk_k=docset.chunk_k
            )
            # get associated vectordb path
            vectordb_folder_path = docset.create_vectordb_name()

            # If vector store folder does not exist, stop
            if path.exists(vectordb_folder_path):
                # create the query chain

                chatquestion = ChatQuestion()
                chatquestion.chat_id = id
                chatquestion.question = question
                chatquestion.answer = ''
                chatquestion.source = ''
                db.session.add(chatquestion)
                db.session.commit()

                querier.make_chain(collection_name, vectordb_folder_path)
                for past_question in past_questions:
                    querier.chat_history.append(HumanMessage(content=past_question.question))
                    querier.chat_history.append(AIMessage(content=past_question.answer))

                result = querier.ask_question(question)
                answer, source = result['answer'], result['source_documents']

                source_ = []
                for document in source:
                    # print(f"Page {document.metadata['page_number']} chunk used: {document.page_content}\n")
                    # metadata: page_number, chunck, source, title, author
                    # page_content
                    tmp = document.metadata
                    tmp['page_content'] = document.page_content
                    source_.append(tmp)

                chatquestion.answer = answer
                chatquestion.source = json.dumps(source_)
                #chatquestion.date_answer = datetime.now().isoformat(sep=' ', timespec='seconds')
                db.session.commit()

                # Update last_used
                chat = Chat.query.filter(Chat.id == id).first()
                chat.last_used = chatquestion.date_answer
                db.session.commit()

                return jsonify(data={'error': False, 'question': question})
            
            else:

                return jsonify(data={'error': True, 'msg': 'The vector databse folder is not found.'})
