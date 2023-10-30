from flask import redirect, flash, url_for, Markup
from flask_login import current_user
from sqlalchemy import text

from flask_app.models import db, Chat, DocSet, UserAuth, UserGroup


def flash_msg(msg, msg_type='danger'):
    flash(Markup('<h3>Invalid permissions</h3><table><tr><td>Reason:</td><td>' + msg + '</td></tr><tr><td>Requested URL:</td><td>request.path</td></tr></table>'), 'danger')


def logged_in(func, *args, **kwargs):
    if current_user and not current_user.is_anonymous:
        return func(*args, **kwargs)
    flash_msg('User is not logged in.')
    return redirect(url_for('invalid_permissions'))


def chat_admin(func=None, *args, **kwargs):
    if current_user and not current_user.is_anonymous:
        if current_user.is_chat_admin:
            if func:
                return func(*args, **kwargs)
            return True
    if func:
        flash_msg('Invalid authorisation.')
        return redirect(url_for('invalid_permissions'))
    return False


def chat(id, func, *args, **kwargs):
    msg = 'Not logged in.'
    if current_user and not current_user.is_anonymous:
        if current_user.is_chat_admin:
            if func:
                return func(*args, **kwargs)
            return True
        if id == 0:
            if func:
                return func(*args, **kwargs)
            return True
        else:
            chat = Chat.query.filter(Chat.id == id).first()
            if chat and chat.user_id == current_user.id:
                if func:
                    return func(*args, **kwargs)
                return True
            msg = 'User \'' + current_user.username + '\' has no authorization for the requested chat.'
    if not func:
        return False
    flash_msg(msg, 'danger')
    return redirect(url_for('invalid_permissions'))


def chats(func, *args, **kwargs):
    if current_user and not current_user.is_anonymous:
        return func(*args, **kwargs)
    flash_msg('No authorization for chats.', 'danger')
    return redirect(url_for('invalid_permissions'))


def my_chats():
    chats = []
    if current_user and not current_user.is_anonymous:
        chats = Chat.query.join(DocSet, Chat.docset_id == DocSet.id) \
                .with_entities(Chat, DocSet) \
                .order_by(Chat.name) \
                .filter(Chat.id >= 1, Chat.user_id == current_user.id) \
                .all()
    return chats

def my_docsets():
    docset_ids_allowed = []
    docsets = DocSet.query.all()
    if current_user and not current_user.is_anonymous:
        if current_user.is_chat_admin:
            docset_ids_allowed = [docset.id for docset in docsets]
        else:
            userauths = UserAuth.query.filter(UserAuth.user_id == current_user.id).all()
            usergroup_ids = [userauth.usergroup_id for userauth in userauths]
            # usergroups = UserGroup.query.filter(UserGroup.id.in_(usergroup_ids)).all()
            if len(usergroup_ids) > 0:
                usergroups = db.session.execute(text('SELECT * FROM ' + UserGroup.__tablename__ + ' WHERE id IN (' + ','.join(map(str, usergroup_ids)) + ');'))
            else:
                usergroups = []
            for usergroup in usergroups:
                for docset in docsets:
                    if getattr(usergroup, 'docset_' + str(docset.id)) == 1:
                        if docset.id not in docset_ids_allowed:
                            docset_ids_allowed.append(docset.id)
    result = []
    for docset in docsets:
        if docset.id in docset_ids_allowed:
            result.append((docset.id, docset.name))
    return result
