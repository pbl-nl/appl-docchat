from pbkdf2 import crypt

from flask import redirect, url_for, flash, current_app
from flask_login import current_user, login_user, logout_user
from flask_ldap3_login import LDAP3LoginManager, AuthenticationResponseStatus
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, PasswordField
from wtforms.validators import DataRequired

from flask_app.models import db, User
from flask_app.helpers import render_chat_template, getSetting

'''

'''

def is_chat_admin_password(password):
    ''' For the chat-admin user, the password should be stored in keypass.
    This password is encryted with (run this in the console):
        print(crypt("the-password-from-keypass"))
    The printed password-hash can then be used to check wether the password matches
    '''
    pwhash = current_app.config.get('CHAT_ADMIN_PW') # '$p5k2$$rJmlSvTO$R5SpqExhyjmlXqC/NTm1IvwRym8mVzkU'
    return pwhash == crypt(password, pwhash)


def login_(request):
    form = loginForm()
    if request.method == 'GET':
        #if current_user and not current_user.is_anonymous:
        #    flash('You are already logged in. Log out? Log is as?', 'danger')
        return render_chat_template('login.html', form=form, current_user=current_user)
    if request.method == 'POST':
        if form.validate_on_submit():
            return redirect(url_for('chats'))
        return render_chat_template('login.html', form=form, current_user=current_user)


def logout_(request):
    form = logoutForm()
    if current_user and not current_user.is_anonymous:
        if request.method == 'GET':
            return render_chat_template('logout.html', form=form)
        if request.method == 'POST':
            logout_user()
            flash('You where logged out.', 'success')
            return redirect(url_for('home'))
    flash('You cannot log out because you are not logged in.', 'danger')
    return render_chat_template('logout.html', form=form)


def loginas_(request):
    form = loginForm()
    if request.method == 'GET':
        return render_chat_template('login.html', form=form, current_user=current_user)
    if request.method == 'POST':
        if form.validate_on_submit():
            return redirect(url_for('chats'))
        return render_chat_template('login.html', form=form, current_user=current_user)


class loginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me', default=False, render_kw={'class': 'yes-checkbox'})
    submit = SubmitField('Submit')

    def validate(form, extra_validators):
        username = form.username.data
        password = form.password.data
        remember_me = form.remember_me.data and getSetting('auto-login')

        if username == 'chat-admin' and is_chat_admin_password(password):
            user = User.query.filter(User.username == 'chat-admin').first()
            flash('Welcome chat-admin', 'success')
            login_user(user, remember_me)
            return True
        elif username == 'guest' and getSetting('guest-allowed'):
            user = User.query.filter(User.username == 'guest').first()
            flash('Welcome quest', 'success')
            login_user(user, False)
            return True
        else:
            'Validate the username/password data against ldap directory'
            ldap = LDAP3LoginManager()
            ldap.init_config({
                'LDAP_PORT': current_app.config['LDAP_PORT'],
                'LDAP_HOST': current_app.config['LDAP_HOST'],
                'LDAP_BIND_DIRECT_CREDENTIALS': False,
                'LDAP_SEARCH_FOR_GROUPS': False,
                'LDAP_USE_SSL': True,
                'LDAP_ALWAYS_SEARCH_BIND': True,
                'LDAP_BASE_DN': current_app.config['LDAP_BASE_DN'],
                'LDAP_USER_DN': current_app.config['LDAP_USER_DN'],
                'LDAP_GROUP_DN': '',
                'LDAP_USER_RDN_ATTR': 'uid',
                'LDAP_USER_LOGIN_ATTR': 'uid',
                'LDAP_BIND_USER_DN': current_app.config['LDAP_BIND_USER_DN'],
                'LDAP_BIND_USER_PASSWORD': current_app.config['LDAP_BIND_USER_PASSWORD']
            })

            result = ldap.authenticate(username, password)
            if result.status == AuthenticationResponseStatus.success:
                user = User.query.filter(User.username == username).first()
                if user:
                    flash('Welcome ' + result.user_info['displayName'], 'success')
                    department = result.user_info['department'].split('.')
                    department.reverse()
                    department = '/'.join(department)
                    if user.name != result.user_info['displayName'] or user.email != result.user_info['mail'] or user.department != department:
                        user.name = result.user_info['displayName']
                        user.email = result.user_info['mail']
                        user.department = department
                        db.session.commit()
                    
                    login_user(user, remember_me)
                    return True
                form.username.errors = ['This username is not known. Please contact the administrator.']
            else:
                form.username.errors = ['Invalid Username/Password.']
                form.password.errors = ['Invalid Username/Password.']
        return False


class logoutForm(FlaskForm):
    submit = SubmitField('Submit')
