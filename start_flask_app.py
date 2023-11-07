from os import path

from flask import current_app, Flask
from flask_login import LoginManager

from flask_app import routes
from flask_app.models import db, Setting, User
from flask_app.background_jobs import background_jobs

if __name__ == '__main__':
    
    app = Flask(__name__,
                template_folder='flask_app\\templates', 
                static_folder='flask_app\\static',
                instance_path=path.join(path.dirname(__file__), 'flask_app', 'instance')
                )

    if not path.exists('flask_app\\config.py'):
        print('Error: Unable to start; flask_app\\config.py not found.\nCopy (or rename) flask_app\\config-example.py to flask_app\\config.py and specify the attributes in the configuration class.')
        exit()

    app.config.from_object('flask_app.config.Config')
    if app.config['ENV'] != 'development' and app.config['ENV'] != 'production':
        print('Error: Unable to start; Invalid ENV specified in flask_app\\config.py.')
        exit()

    print('Starting NMDC chat:\nEnvironment:\t\t\t' + app.config['ENV'] + '\nDatabase connection string:\t' + app.config['SQLALCHEMY_DATABASE_URI'], flush=True)

    app.app_context().push()
    
    db.init_app(app)
    db.create_all()

    background_jobs.init_app(app)
    
    # Set required settings if they do not exists, or update (without value) if they do exist.
    defaults = [
            #('origin', 'key', 'label', 'value', 'help')
            ('boolean',	'remember-me', 'Remember me allowed', '1', 'Allow user to check \'Remember me\' so the user stays logged in during different browser sessions.'),
            ('boolean', 'guest-allowed', 'Guest login allowed', '0', 'Display a \'Guest login\' button for anonimeous login. Remark: All users that log in will \'share\' the user with username \'guest\'.'),
            ('boolean', 'guest-auto-login', 'Guests login automatically', '0', 'If no user is logged in (remember-me), the guest is logged in automatically (if allowed: guest-allowed)'),
    ]

    for default in defaults:
        setting = Setting.query.filter(Setting.setting == default[1]).first()
        if not setting:
            setting = Setting()
            setting.setting_type, setting.setting, setting.label, setting.value, setting.info = default
            db.session.add(setting)
            db.session.commit()
        else:
            setting.setting_type = default[0]
            setting.label = default[2]
            setting.info = default[4]
            db.session.commit()

    # Add default users if they do not exists
    guest = User.query.filter(User.username == 'guest').first()
    if not guest:
        guest = User()
        guest.username = 'guest'
        guest.name = 'Guest'
        guest.email = ''
        guest.is_chat_admin = False
        db.session.add(guest)
        db.session.commit()
        
    admin = User.query.filter(User.username == 'chat-admin').first()
    if not admin:
        admin = User()
        admin.username = 'chat-admin'
        admin.name = 'Chat administrator'
        admin.email = ''
        admin.is_chat_admin = True
        db.session.add(admin)
        db.session.commit()
    
    # 
    login_manager = LoginManager()
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.filter(User.id == user_id).first()


    
    routes.init_app(app)

    if current_app.config.get('ENV') == 'production':
        # app.run(host='0.0.0.0', debug = False)
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000, threads=50)
    else:
        app.run(debug=False)
