from pbkdf2 import crypt


class Config:
    # Flask config variables
    ENV = 'development'
    SECRET_KEY = 'some-secret-key'

    # Database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///chat.sqlite?check_same_thread=False'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask-User
    USER_APP_NAME = 'chatNMDC'
    USER_ENABLE_EMAIL = False
    USER_ENABLE_USERNAME = True
    USER_REQUIRE_RETYPE_PASSWORD = False

    # Chat admin
    CHAT_ADMIN_PW = crypt('your-chat-admin-password')

    # Flask-wtf
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = 'some-other-secret-key'

    
    # LDAP
    LDAP_PORT = 0
    LDAP_HOST = 'ldap-host'
    LDAP_BASE_DN = 'dc=organisation,dc=nl'
    LDAP_USER_DN = 'ou=Users,ou=Accounts,ou=organisation'
    LDAP_BIND_USER_DN = 'ldap-user'
    LDAP_BIND_USER_PASSWORD = 'ldap-user-password'
