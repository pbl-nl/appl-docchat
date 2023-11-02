import re

from flask import flash, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import HiddenField, StringField, BooleanField, SubmitField
from wtforms.validators import Length, ValidationError

from flask_app.models import db, User, UserAuth, UserGroup
from flask_app.models.permissions import chat_admin
from flask_app.helpers import render_chat_template


'''
This form contains all for inserting, updating and deleting a user
'''


class UserForm(FlaskForm):

    # Field definitions
    id = HiddenField('ID')
    username = StringField('Username', default='', validators=[Length(min=6, max=32)])
    is_chat_admin = BooleanField('Chat administrator', default=False, render_kw={'class': 'yes-checkbox'})

    docsets = StringField('Document sets')
    email = StringField('E-mail', default='', render_kw={'disabled': 'disabled'}, validators=[Length(min=0, max=64)])
    department = StringField('Department', render_kw={'disabled': 'disabled'}, default='', validators=[Length(min=0, max=64)])
    name = StringField('Name', default='', validators=[Length(min=3, max=64)])
    user_auth = StringField('Auths')
    submit = SubmitField('Opslaan')

    # Custom validation    ( See: https://wtforms.readthedocs.io/en/stable/validators/ )

    def validate_username(form, field):
        if not re.search(r'^[a-zA-Z0-9-_]+$', field.data):
            raise ValidationError('Invalid name; Only letters, digits, - _ characters allowed.')
        else:
            same_username = User.query.filter(User.username == field.data.strip(), User.id != form.user_id_for_validation).all()
            if len(same_username) >= 1:
                raise ValidationError('This username already exists.')


    # Handle the request (from routes.py) for this form
    def handle_request(self, method, id):

        # Show the form
        usergroups = UserGroup.query.all()
        if id > 0:
            # Get all authorisations (= groups where user is a member of) for this user
            userauths = UserAuth.query.filter(UserAuth.user_id == id).all()

            for usergroup in usergroups:
                setattr(usergroup, 'checked', '')
                for usertauth in userauths:
                    # if the user is member of this user group: checked
                    if usertauth.usergroup_id == usergroup.id:
                        setattr(usergroup, 'checked', ' checked="checked"')
        else:
            # New users are not member of any group
            for usergroup in usergroups:
                setattr(usergroup, 'checked', '')
        if method == 'GET':
            if id > 0:
                # Get record from database and set the form values (if id == 0 the defaults are used)
                obj = User.query.get(id)
                obj.fields_to_form(self)

            # Show the form
            return render_chat_template('user.html', form=self, usergroups = usergroups)
        
        # Save the form
        if method == 'POST':
            self.user_id_for_validation = id
            if self.validate():
                if id >= 1:
                    # The table needs to be updated with the new values
                    obj = User.query.get(id)
                    obj.fields_from_form(self)
                    db.session.commit()

                else:
                    # A new record must be inserted in tyhe table
                    obj = User()
                    obj.fields_from_form(self)
                    db.session.add(obj)
                    db.session.commit()
                    id = obj.id
                
                # Handle the authorisation
                # Use existing records, insert record if nescesary and delete
                # unused records.
                userauths = UserAuth.query.filter(UserAuth.user_id == id).all()
                usergroup_ids, userauth_ids = [], [userauth.id for userauth in userauths]
                for usergroup in usergroups:
                    if getattr(self, 'usergroup_' + str(usergroup.id)):
                        usergroup_ids.append(usergroup.id)
                i, keep = 0, 0
                for userauth in userauths:
                    if i < len(usergroup_ids):
                        # Use existing record
                        userauth.usergroup_id = usergroup_ids[i]
                        usergroup_ids[i] = -1
                        keep += 1
                    else:
                        # Delete unused record
                        db.session.delete(userauth)
                    i += 1
                # Insert new records
                for i in range(len(usergroup_ids)):
                    if usergroup_ids[i] > 0:
                        obj = UserAuth()
                        obj.user_id = id
                        obj.usergroup_id = usergroup_ids[i]
                        db.session.add(obj)
                db.session.commit()

                flash('The user is saved.', 'info')
                return redirect(url_for('users'))
            
            # Validation failed: Show the form with the errors
            return render_chat_template('user.html', form=self, usergroups = usergroups)

        # Delete the user
        if method == 'DELETE':
            user = User.query.filter(User.id == id).first()
            if user.username != 'chat-admin' and user.username != 'guest':
                db.session.execute(UserAuth.__table__.delete().where(UserAuth.user_id == id))
                User.query.filter(User.id == id).delete()
                db.session.commit()
                flash('The user has been deleted.', 'info')
            else:
                flash('The user ' + user.username + ' canno\'t be deleted.', 'danger')
            return redirect(url_for('users'))
