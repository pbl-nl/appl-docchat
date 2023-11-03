function page_ready() {
    if (parseInt($('#id').val()) >= 1) {
        $('form').find('input,select').each(function(i, el) {
            var id = $(el).prop('id');
            if (id != 'id' && id != 'csrf_token') {
                $(el).attr('disabled', 'disabled');
            }
        });
        $('#name,.btn-submit').removeAttr('disabled');
        var dz = $('#dropzone').dropzone({ 
            url: '/docset-upload-file/' + $('#id').val(),
            maxFiles: 1,
            withCredentials: true,
            maxFilesize: 16384,
            chunking: true,
            forceChunking: true,
            chunkSize: 500000,
            retryChunks: true,
            retryChunksLimit: 3,
            parallelChunkUploads: false, // Very important! If set to true, the upload on larger files fails.
            renameFile: function(f) {
                return f.lastModified+'-'+f.name;
            },
            init: function() {
                var that = this;
                that.on('uploadprogress', function(file, progress, bytesSent) {
                    if (progress == 100) {
                        $(file.previewElement).find('.dz-progress').css('opacity', 0);
                        $(file.previewElement).find('.dz-success-mark').css({'opacity': 1, 'margin': '22px 0 0 -42px', 'font-size': '20px', 'color': 'black'}).addClass('blink').html('Chunking');
                    }
                });
                that.on('success', function(file, response) {
                    // remove visible file/element in dropzone container
                    that.removeFile(file);
                    if (that.files.length == 0) { // All files are uploaded
                        chat_dialog('Upload', 'Upload completed.',['View chunks', function() {
                                url = location.origin + '/docset-files/' + $('#id').val();
                                location.assign(url);
                            }], ['Ok', function(){
                                location.reload();
                            }]);
                    }
                });
                that.on('error', function(file, response) {
                    // remove visible file/element in dropzone container
                    that.removeFile(file);
                    chat_dialog('Upload', 'Error during upload.',['Ok', function() {}]);
                });
            },
        }).addClass('dropzone');
    }
}

function docset_delete_file(file_id) {
    msg='Delete file \'' + $('#file_name_' + file_id).html() + '\'?';
    url = location.origin + '/docset-delete-file/' + file_id;
    chat_delete(msg, url);
}
