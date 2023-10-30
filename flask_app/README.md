# Chat NMDC web-application
A web-application which offers:
1. Create document sets:
   * Specify the model parameters
   * Upload (new) documents
   * See how the chunks (and their overlap) look like
   * Filter chunks by keyword
2. Start a chat on a specific dosument set.
3. View the sources on which an answer is based
   * The used sources are highlighted
   * See how the chunks (and their overlap) look like
   * Filter chunks by keyword4. User management and authorisation
4. User management:
   * Default chat-admin and guest user
   * User management
   * Authorisation:
     * User can be part of user-groups
     * User-groups have authorisation on document sets
   * Login via LDAP

The web-appliaction works with a database (default sqlite, others possible) and can be installed locally or on a server.

## Installation
Setup the configuration
1. Setup the underlaying system (appl-docchat) and make sure it's working
2. Rename /flask_app/config-example.py to /flask_app/config.py
3. Change the settings in config.py, at least:
   * SECRET_KEY = 'some-secret-key'
   * CHAT_ADMIN_PW = crypt('your-chat-admin-password')
   * WTF_CSRF_SECRET_KEY = 'some-other-secret-key'
4. Optionally:
   * Change the name of the sqlite file, or specify connectionstring for MySQL, ...
   * Specify LDAP configuration to allow users to validate againt the Active Directory.

## Get started
To quickly get started:
1. Follow the steps from *Installation*
2. Run:<br>
<code>python start_flask_app.py</code>
3. When the message **Running on http://127.0.0.1:5000 Press CTRL+C to quit** appears:
   * Open the URL (it may be a different one) in your browser
   * Log in as chat-admin and the password set in the config.py
   * Create a document set
   * Create a chat
   * Start chatting

## Relation to appl-docchat
The Chat NMDC web-application is build on top of appl-docchat. It uses:
1. The Ingester class to chunk uploaded files.
2. The Querier class to chat.
3. The docs folder to store uploaded files
4. The vector_stores folder to store vector dabases
It is possible to use the base-scripts, the streamlit app and the Chat NMDC web-application alogside each other, although the latter does not 'see' the folders created with the base-scripts.

## User and user group management
### Users
By default, the chat-administrator and a guest user exist. You can add users, who can login if you
arrange the LDAP parameters in the config file. The LDAP-server authenticates the user who wants to
log in. If the LDAP-server 'knows' the user then the user is logged in, with the authorisation
specified per user-group.
### User groups
User groups are used to grant access (or not) to document sets. If a user is member of a user group,
the user is authorised for all accessable document sets in the user group. Users can be member of
zero or more user groups.

## Document sets
When a document set is created, the model parameters which will be used to make the documents 'chat-
able' are stored. They cannot be changed afterwards, because one or more documents in the set may
have been prepared conform the model parameters.

## Known issues
1. The deletion of a file does not delete the chunks (documents) from the vector database.
2. Login as does not work properly when guest users are logged in automatically

## To Do
1. The basic functionality for guest users has been implemented, but:
   * When a few people use the guest user account, they will see each others chats.
   * When a lots of people use the guest account, who knows what will happen.
   * What has to be made is creation of a user:
     * who is a copy of the guest user (inherits the authorisation)
     * use (part of) the flask-session as the username
     * the user has to be deleted when the session no longer exists
2. The document set is actually a document & model set. It would be nice to manage the document set and model (parameters) seperately.
3. Create a proper backend, so uploaded files can be chuncked in the background:
   * Create a backend based on SQLAlchemy (communication via database)
   * Do not chunk after upload, but post a job
   * Change the frontend; update frontend while chunking takes place
4. Chat download: What is the purpose? Remove or make it usefull
5. Change docset_? column names in UserGroup to proper normalised DB & ORM