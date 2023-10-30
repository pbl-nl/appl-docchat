from flask import flash, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import HiddenField, StringField, SubmitField
from wtforms.validators import Length, ValidationError

from flask_app.models import db, Setting
from flask_app.models.permissions import chat_admin
from flask_app.helpers import render_chat_template


'''
This form contains all for updating a setting
'''


class SettingForm(FlaskForm):

    # Field definitions
    id = HiddenField('ID')
    setting_type = StringField('Setting', render_kw={'class': 'dont_show_as_input', 'disabled': 'disabled'})
    setting = StringField('Setting', render_kw={'class': 'dont_show_as_input', 'disabled': 'disabled'})
    label = StringField('Label', render_kw={'class': 'dont_show_as_input', 'disabled': 'disabled'})
    value = StringField('Waarde', [Length(min=0, max=256)], render_kw={'size': '60'})
    info = HiddenField('Info')
    submit = SubmitField('Opslaan')

    # Custom validation    ( See: https://wtforms.readthedocs.io/en/stable/validators/ )

    def validate_username(form, field):
        if form.value.data.strip() == '':
            raise ValidationError('Invalid value')


    # Handle the request (from routes.py) for this form
    def handle_request(self, method, id):

        # Show the form
        if method == 'GET':
            if id > 0:
                # Get record from database and set the form values (if id == 0 the defaults are used)
                obj = Setting.query.get(id)
                obj.fields_to_form(self)

            # Show the form
            return render_chat_template('setting.html', form=self)
        
        # Save the form
        if method == 'POST':
            if self.validate():
                if id >= 1:
                    # The table needs to be updated with the new values
                    obj = Setting.query.get(id)
                    obj.fields_from_form(self)
                    db.session.commit()

                flash('The setting is saved.', 'info')
                return redirect(url_for('settings'))
            
            # Validation failed: Show the form with the errors
            return render_chat_template('setting.html', form=self)

