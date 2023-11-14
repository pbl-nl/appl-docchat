
import json
from time import sleep

from flask import request, flash, jsonify, current_app
from flask_login import current_user, login_user

import flask_app.models.permissions as permission
from flask_app.login import login_, logout_, loginas_
from flask_app.forms.user import UserForm
from flask_app.forms.usergroup import UserGroupForm
from flask_app.forms.docset import DocSetForm
from flask_app.forms.setting import SettingForm
from flask_app.forms.chat import ChatForm
from flask_app.models import db, User, UserGroup, DocSet, Setting, Job
from flask_app.helpers import render_chat_template, getSetting


def auto_login():
    logged_in = True if current_user and not current_user.is_anonymous else False
    if not logged_in:
        # Check settings
        # if remember me, check if there is u user to be rememberred
        # if not remember me, check if the geust-user must be logged in automatically
        remember_me, guest_allowed, guest_auto_login = getSetting('remember-me'), getSetting('guest-allowed'), getSetting('guest-auto-login')
        if guest_allowed and guest_auto_login:
            # Login as quest
            user = User.query.filter(User.username == 'guest').first()
            flash('You where automatically logged in as guest', 'success')
            login_user(user, remember_me)

def init_app(app):
    '''
    General routes, no permissions required
    '''
    @app.route('/')
    def home():
        auto_login()
        return render_chat_template('home.html')

    @app.route('/invalid_permissions')
    def invalid_permissions():
        return render_chat_template('invalid_permissions.html')

    @app.route('/help')
    def help():
        auto_login()
        return render_chat_template('help.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        return login_(request)

    @app.route("/logout", methods=['GET', 'POST'])
    def logout():
        return logout_(request)

    @app.route("/loginas", methods=['GET', 'POST'])
    def loginas():
        return loginas_(request)


    '''
    Chat related routes
    '''
    @app.route('/chat/<int:id>', methods=['GET', 'POST'])
    def chat(id):
        auto_login()
        form = ChatForm()
        return permission.chat(id, form.handle_request, request.method, id)

    @app.route('/chats', methods=['GET'])
    def chats():
        auto_login()
        chats = permission.my_chats()
        return permission.chats(render_chat_template, 'chats.html', chats=chats)
    
    @app.route('/chat/delete/<int:id>', methods=['GET'])
    def chat_delete(id):
        auto_login()
        form = ChatForm()
        return permission.chat(id, form.handle_request, 'DELETE', id)

    @app.route('/chat/download/<int:id>', methods=['GET'])
    def chat_download(id):
        auto_login()
        form = ChatForm()
        return permission.chat(id, form.handle_request, 'DOWNLOAD', id)

    @app.route('/chat/clear/<int:id>', methods=['GET'])
    def chat_clear(id):
        auto_login()
        form = ChatForm()
        return permission.chat(id, form.handle_request, 'CLEAR', id)

    @app.route('/question/<int:id>', methods=['POST'])
    def question(id):
        auto_login()
        form = ChatForm()
        if permission.chat(id, False):
            return form.handle_request('QUESTION', id, {'question': request.form['question']})
        return jsonify(data={'error': True, 'message': 'Invalid authorisation'})


    '''
    Administrator related routes
    '''

    @app.route('/users')
    def users():
        users = User.query.order_by(User.username).all()
        return permission.chat_admin(render_chat_template, 'users.html', users=users)

    @app.route('/user/<int:id>', methods=['GET', 'POST'])
    def user(id):
        form = UserForm()
        if request.method == 'POST':
            # Retrieve variable fields, place values in form
            usergroups = UserGroup.query.all()
            for usergroup in usergroups:
                columnname = 'usergroup_' + str(usergroup.id)
                usergroup_value = False
                try:
                    if str(request.form[columnname]) == '1':
                        usergroup_value = True
                except:
                    pass
                setattr(form, 'usergroup_' + str(usergroup.id), usergroup_value)
        return permission.chat_admin(form.handle_request, request.method, id)

    @app.route('/user/delete/<int:id>', methods=['GET'])
    def user_delete(id):
        form = UserForm()
        return permission.chat_admin(form.handle_request, 'DELETE', id)

    # Usergroups
    @app.route('/usergroups')
    def usergroups():
        usergroups = UserGroup.query.order_by(UserGroup.name).all()
        return permission.chat_admin(render_chat_template, 'usergroups.html', usergroups=usergroups)

    @app.route('/usergroup/<int:id>', methods=['GET', 'POST'])
    def usergroup(id):
        form = UserGroupForm()
        # See app/forms/usergroup.py for an explanation
        if request.method == 'POST':
            # Retrieve variable fields, place values in form
            docsets = DocSet.query.all()
            for docset in docsets:
                columnname = 'docset_' + str(docset.id)
                docset_value = False
                try:
                    if str(request.form[columnname]) == '1':
                        docset_value = True
                except:
                    pass
                setattr(form, 'docset_' + str(docset.id), docset_value)
        return permission.chat_admin(form.handle_request, request.method, id)

    @app.route('/usergroup/delete/<int:id>')
    def usergroup_delete(id):
        form = UserGroupForm()
        return permission.chat_admin(form.handle_request, 'DELETE', id)

    # Docsets
    @app.route('/docsets')
    def docsets():
        docsets = DocSet.query.order_by(DocSet.name).all()
        return permission.chat_admin(render_chat_template, 'docsets.html', docsets=docsets)

    @app.route('/docset/<int:id>', methods=['GET', 'POST'])
    def docset(id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, request.method, id)

    @app.route('/docset-files/<int:id>')
    def docset_files(id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, 'FILES', id)

    @app.route('/docset-chunks/<int:id>')
    def docset_chunks(id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, 'CHUNKS', id)

    @app.route('/docset-upload-file/<int:id>', methods=['POST'])
    def docset_upload_file(id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, 'UPLOAD-FILE', id)

    @app.route('/docset-delete-file/<int:id>/<int:file_id>')
    def docset_delete_file(id, file_id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, 'DELETE-FILE', id, file_id)

    @app.route('/docset-delete/<int:id>')
    def docset_delete(id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, 'DELETE', id)

    @app.route('/docset-status/<int:id>', methods=['POST'])
    def docset_status(id):
        form = DocSetForm()
        return permission.chat_admin(form.handle_request, 'STATUS', id)

    # Miscellaneous
    @app.route('/jobs')
    def jobs():
        jobs = Job.query.order_by(Job.id).all()
        return permission.chat_admin(render_chat_template, 'jobs.html', jobs=jobs)

    @app.route('/settings')
    def settings():
        settings = Setting.query.all()
        return permission.chat_admin(render_chat_template, 'settings.html', settings=settings, config=current_app.config)

    @app.route('/setting/<int:id>', methods=['GET', 'POST'])
    def setting(id):
        form = SettingForm()
        return permission.chat_admin(form.handle_request, request.method, id)
