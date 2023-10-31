import re
from flask import url_for, redirect, flash
from flask_wtf import FlaskForm
from wtforms import HiddenField, StringField, SubmitField
from wtforms.validators import Length, ValidationError
from sqlalchemy import text

from flask_app.models import db, DocSet, UserGroup, UserAuth
from flask_app.helpers import render_chat_template


'''
Explanation how the authorisation on specific docsets works


*** The file app.forms.docset.py ***
When docsets are added & deleted, the variable fields are set in the database by
the function DocSet.setUserGroupFields(). This ensures that the user-group-table
has a 'docset_?' present for each document set.

*** The file app.routes.py ***
The route 'usergroup/<id>' adds the values from the variable fields to the form in case
the request is a POST-request (save the form).

*** This file ***
The values from variable fields in this form are stored to the database after handling
a 'normal' save.
On a 'GET' request the form is expanded with values from the database (or default False
if it is a new usergroup) and the document-sets. The True values create a 'checked="checked"'
value so the form can reflect the database-situation.

*** The file app/templates/usergroup.html ***
The authorisation checkboxes (and their labels) are generated using the expanded data of
the form (docsets and columnnames)
'''


'''
This form contains all for inserting, updating and deleting a usergroup
'''


class UserGroupForm(FlaskForm):

    # Field definitions
    id = HiddenField('ID', default=0)
    name = StringField('Name', default='', validators=[Length(min=3, max=64)])
    submit = SubmitField('Save')


    # Custom validation    ( See: https://wtforms.readthedocs.io/en/stable/validators/ )
    def validate_path(form, field):
        if not re.search(r'^[a-z0-9-]+$', field.data):
            raise ValidationError('Invalid name; Only lowercase, digits and - allowed.')


    # Handle the request (from routes.py) for this form
    def handle_request(self, method, id):

        docsets = DocSet.query.all()
        columnnames = []
        for docset in docsets:
            colunmname = 'docset_' + str(docset.id)
            columnnames.append(colunmname)

        # Show the form
        if method == 'GET':
            if id > 0:
                # Get record from database and set the form values (if id == 0 the defaults are used)
                obj = UserGroup.query.get(id)
                obj.fields_to_form(self)
                # Set variable fields
                query = 'SELECT ' + ','.join(columnnames) +' FROM ' + UserGroup.__tablename__ + ' WHERE id = ' + str(id) + ';'
                i, records = 0, db.session.execute(text(query)).fetchall()
                for columnname in columnnames:
                    setattr(self, columnname, ' checked="checked"' if records[0][i] else '')
                    i += 1
            else:
                # Set variable fields
                for columnname in columnnames:
                    setattr(self, columnname, '')

            # Show the form
            return render_chat_template('usergroup.html', form=self, docsets=docsets, columnnames=columnnames)
        
        # Save the form
        if method == 'POST':
            if self.validate():
                if id >= 1:
                    # The table needs to be updated with the new values
                    obj = UserGroup.query.get(id)
                    obj.fields_from_form(self)
                    db.session.commit()

                else:
                    # A new record must be inserted in tyhe table
                    obj = UserGroup()
                    obj.fields_from_form(self)
                    db.session.add(obj)
                    db.session.commit()
                    id = obj.id
                
                # Updata database with variable fields
                query = 'UPDATE ' + UserGroup.__tablename__ + ' SET ' + ', '.join([columnname + ' = ' + str(getattr(self, columnname)) for columnname in columnnames]) + ' WHERE id = ' + str(id) + ';'
                db.session.execute(text(query))
                db.session.commit()

                flash('The usergroup is saved.', 'info')
                return redirect(url_for('usergroups'))
            
            # Validation failed: Show the form with the errors
            return render_chat_template('usergroup.html', form=self)

        # Delete the usergroup
        if method == 'DELETE':
            UserAuth.query.filter(UserAuth.usergroup_id == id).delete()
            UserGroup.query.filter(UserGroup.id == id).delete()
            db.session.commit()
            flash('The usergroup has been deleted.', 'info')
            return redirect(url_for('usergroups'))

