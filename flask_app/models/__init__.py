from os import path
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.sql import func

from flask_login import current_user


db = SQLAlchemy()


class Chat(db.Model):
    id = Column(Integer, primary_key=True, nullable=False)
    user_id = Column(Integer, default=0, nullable=False)
    name = db.Column(db.String(64), default='', nullable=False)
    docset_id = db.Column(Integer, default=0, nullable=False)
    last_used = Column(DateTime, default=func.now())

    def fields_from_form(self, form):
        self.user_id = current_user.id
        self.name = form.name.data
        self.docset_id = form.docset_id.data

    def fields_to_form(self, form):
        form.id.data = self.id
        form.name.data = self.name
        form.docset_id.data = self.docset_id


class ChatQuestion(db.Model):
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=func.now())
    date_answer = Column(DateTime, onupdate=func.now(), nullable=True)
    chat_id = Column(Integer, nullable=False)
    question = db.Column(db.String(256), nullable=False)
    answer = db.Column(db.String(4096), nullable=False)
    source = db.Column(db.String(4096), nullable=False)

class DocSet(db.Model):
    id = Column(Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    llm_type = db.Column(db.String(32), nullable=False)
    llm_modeltype = db.Column(db.String(32), nullable=False)
    embeddings_provider = db.Column(db.String(32), nullable=False)
    embeddings_model = db.Column(db.String(32), nullable=False)
    text_splitter_method = db.Column(db.String(64), nullable=False)
    chain = db.Column(db.String(32), nullable=False)
    chain_type = db.Column(db.String(32), nullable=False)
    chain_verbosity = db.Column(db.Boolean, nullable=False)
    search_type = db.Column(db.String(32), nullable=False)
    vecdb_type = db.Column(db.String(32), nullable=False)
    chunk_size = db.Column(Integer, nullable=False)
    chunk_overlap = db.Column(Integer, nullable=False)
    chunk_k = db.Column(Integer, nullable=False)

    def get_doc_path(self):
        return path.join('./docs', 'docset_' + str(self.id))

    def create_vectordb_name(self):
        vectordb_name = "_" + self.vecdb_type + "_" + str(self.chunk_size) + "_" + str(self.chunk_overlap) + "_" + self.embeddings_provider
        return path.join('./vector_stores', 'docset_' + str(self.id) + vectordb_name)

    def get_collection_name(self):
        return 'docset_' + str(self.id)

    def fields_from_form(self, form):
        self.name = form.name.data
        self.llm_type = form.llm_type.data
        self.llm_modeltype = form.llm_modeltype.data
        self.embeddings_provider = form.embeddings_provider.data
        self.embeddings_model = form.embeddings_model.data
        self.text_splitter_method = form.text_splitter_method.data
        self.chain = form.chain.data
        self.chain_type = form.chain_type.data
        self.chain_verbosity = form.chain_verbosity.data
        self.search_type = form.search_type.data
        self.vecdb_type = form.vecdb_type.data
        self.chunk_size = form.chunk_size.data
        self.chunk_overlap = form.chunk_overlap.data
        self.chunk_k = form.chunk_k.data

    def fields_to_form(self, form):
        form.id.data = self.id
        form.name.data = self.name
        form.llm_type.data = self.llm_type
        form.llm_modeltype.data = self.llm_modeltype
        form.embeddings_provider.data = self.embeddings_provider
        form.embeddings_model.data = self.embeddings_model
        form.text_splitter_method.data = self.text_splitter_method
        form.chain.data = self.chain
        form.chain_type.data = self.chain_type
        form.chain_verbosity.data = self.chain_verbosity
        form.search_type.data = self.search_type
        form.vecdb_type.data = self.vecdb_type
        form.chunk_size.data = self.chunk_size
        form.chunk_overlap.data = self.chunk_overlap
        form.chunk_k.data = self.chunk_k


class DocSetFile(db.Model):
    id = Column(Integer, primary_key=True)
    docset_id = Column(Integer, nullable=False)
    no = Column(Integer, nullable=False)
    filename = db.Column(db.String(32), nullable=True)


class User(db.Model):
    id = Column(Integer, primary_key=True)
    username = db.Column(db.String(32), nullable=True)
    name = db.Column(db.String(64), nullable=True)
    email = db.Column(db.String(64), nullable=True)
    department = db.Column(db.String(64), nullable=True)
    is_chat_admin = db.Column(db.Boolean())
    is_authenticated = True
    is_active = True
    is_anonymous = False

    def get_id(self):
        return self.id

    def fields_from_form(self, form):
        self.username = form.username.data
        self.name = form.name.data
        self.email = form.email.data
        self.is_chat_admin = form.is_chat_admin.data

    def fields_to_form(self, form):
        form.id.data = self.id
        form.username.data = self.username
        form.name.data = self.name
        form.email.data = self.email
        form.is_chat_admin.data = self.is_chat_admin


class UserGroup(db.Model):
    id = Column(Integer, primary_key=True)
    name = db.Column(db.String(32), default='', nullable=False)

    def fields_from_form(self, form):
        self.name = form.name.data

    def fields_to_form(self, form):
        form.id.data = self.id
        form.name.data = self.name


class UserGroupDocSet(db.Model):
    id = Column(Integer, primary_key=True)
    docset_id = Column(Integer, nullable=True)
    usergroup_id = Column(Integer, nullable=True)
    
        
class UserAuth(db.Model):
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    usergroup_id = Column(Integer, nullable=True)

class Setting(db.Model):
    id = Column(Integer, primary_key=True)
    setting_type = db.Column(db.String(16), nullable=True)
    setting = db.Column(db.String(256), nullable=True)
    label = db.Column(db.String(64), nullable=True)
    value = db.Column(db.String(256), nullable=True)
    info = db.Column(db.String(8192), nullable=True)

    def fields_from_form(self, form):
        if self.setting_type == 'boolean':
            self.value = True if form.value and form.value.data == '1' else False
        else:
            self.value = form.value.data

    def fields_to_form(self, form):
        form.id.data = self.id
        form.setting_type.data = self.setting_type
        form.setting.data = self.setting
        form.label.data = self.label
        form.info.data = self.info
        if self.setting_type == 'boolean':
            form.value.data = True if self.value == '1' or self.value == 1 or self.value == True else False
        else:
            form.value.data = self.value


class Job(db.Model):
    id = Column(Integer, primary_key=True)
    bind_to_id = Column(Integer, default=0)
    dt = Column(DateTime, default=func.now())
    dt_last = Column(DateTime, default=func.now(), onupdate=func.now())
    status_system = db.Column(db.String(16), nullable=True)
    status = db.Column(db.String(16), nullable=True)
    status_msg = db.Column(db.String(128), nullable=True)
    job_type = db.Column(db.String(16), nullable=True)
    job_parms = db.Column(db.String(1048), nullable=True)
