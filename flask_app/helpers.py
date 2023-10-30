from os import path, remove, utime

from flask import current_app, Markup, render_template, request, jsonify, send_file
from flask_login import current_user

from ingest.ingester import Ingester
from flask_app.models import db, DocSetFile, Setting


def form_fields_from_object(form, obj, fields):
    for field in fields:
        form[field].data = getattr(obj, field)


def object_fields_from_form(obj, form, fields):
    for field in fields:
        setattr(obj, field, getattr(getattr(form, field),'data'))


def getSetting(machine_name):
    setting_ = Setting.query.filter(Setting.setting == machine_name).first()
    if setting_:
        if setting_.setting_type == 'boolean':
            v = str(setting_.value).lower()
            return True if v == '1' or v == 'true' else False
        return setting_.value
    return None


def render_chat_template(*args, **kwargs):
    logged_in = True if current_user and not current_user.is_anonymous else False
    is_chat_admin = True if logged_in and current_user.is_chat_admin else False
    is_home = True if args[0] == 'home.html' else False
    loginas = False
    if logged_in:
        username = current_user.name
        if getSetting('guest-allowed') and getSetting('guest-auto-login'):
            loginas = True
    else:
        username = 'Nobody'
    return render_template(logged_in=logged_in, is_chat_admin=is_chat_admin, is_home=is_home, loginas=loginas, username=username, *args, **kwargs)


def size_to_human(size):
    if size < 1024:
        return str(size) + ' b'
    if size < 1024 * 1024:
        return str(round(size / 1024, 1)) + ' Kb'
    if size < 1024 * 1024 * 1024:
        return str(round(size / 1024 / 1024, 1)) + ' Mb'
    if size < 1024 * 1024 * 1024 * 1024:
        return str(round(size / 1024 / 1024 / 1024, 1)) + ' Gb'
    return str(round(size / 1024 / 1024 / 1024 / 1024, 1)) + ' Tb'


def upload_file(docset):
    to_path = docset.get_doc_path()
    file = request.files.get('file')
    filename = path.basename(file.filename.replace('\\', '/'))

    allowed_extensions = ['pdf']

    dot, ext = filename.rfind('.'), ''
    if dot >= 0:
        ext = filename[dot + 1:].lower()
    if not ext in allowed_extensions:
        return False, jsonify({'error': True, 'msg': 'The extension \'' + ext + '\' is not allowed.'})
    pos = filename.find('-')
    if pos >= 1:
        dt = filename[0:pos]
        dt = int(dt)
        filename = filename[pos+1:]
    else:
        dt = False
    current_chunk = int(request.form['dzchunkindex'])
    uuid = request.form['dzuuid']

    to_file = path.join(to_path, filename)
    
    if current_chunk == 0 and path.isfile(to_file):  
        remove(to_file)
    with open(to_file, 'ab+') as f:
        f.seek(int(request.form['dzchunkbyteoffset']))
        f.write(file.stream.read())
        
    if current_chunk == int(request.form['dztotalchunkcount']) - 1:
        status = 'Upload complete'
        '''
        if dt:
            utime(to_file, (dt, dt))
        '''

        files = DocSetFile.query.filter(DocSetFile.docset_id == docset.id).all()
        if files:
            file_no = len(files) + 1
        else:
            file_no = 1
        docsetfile = DocSetFile()
        docsetfile.docset_id = docset.id
        docsetfile.no = file_no
        docsetfile.filename = filename
        db.session.add(docsetfile)
        db.session.commit()
        ingest(docset, filename, file_no)
    else:
        status = 'Uploading chunk ' + str(1 + current_chunk) + '/' + request.form['dztotalchunkcount']
    
    return jsonify({'id': docset.id, 'status': status})


def ingest(docset, filename, file_no):
    ingester = Ingester(
        docset.get_collection_name(), 
        path.join(docset.get_doc_path(), filename),
        docset.create_vectordb_name(), 
        embeddings_provider=docset.embeddings_provider, 
        embeddings_model=docset.embeddings_model, 
        vecdb_type=docset.vecdb_type,
        chunk_size=docset.chunk_size,
        chunk_overlap=docset.chunk_overlap,
        file_no=file_no
    )
    ingester.ingest()
