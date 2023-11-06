var status_polling=false;

function pol_status() {
    status_polling = true;
    var url = location.origin + '/docset-status/' + $('#id').val();
    $.ajax({
        type: "POST",
        url: url,
        success: function (data) {
            if (data.error) {
                chat_dialog('Error', data.msg, ['Ok', function() {window.location.reload();}]);
            } else {
                var i, file, tr;
                var pol_again = false, pol_count = 0;
            
                $('[id^=file-tr-]').addClass('mark-as-deleted');
                for (i = 0; i < data.files.length; i++) {
                    file = data.files[i];
                    tr = $('#file-tr-' + file.id);
                    if (tr.length == 0) {
                        $('#file-table').append('<tr id="file-tr-' + file.id + '"><td id="f-' + file.id + '"></td><td id="file_name_' + file.id + '">' + file.filename + '</td><td style="white-space: nowrap;">' + file.dt + '</td><td>' + file.size + '</td><td id="status-' + file.id + '"></td><td><a href="javascript:docset_delete_file(' + file.id + ');" class="btn btn-secondary btn-sm">Delete</a></td></tr>');
                        tr = $('#file-tr-' + file.id);
                    }
                    // Set F column
                    $('#f-' + file.id).html('F' + file.no);
                    // Set Status column
                    $('#status-' + file.id).html(file.status);
                    if (file.status == 'Chunking') {
                        $('#status-' + file.id).addClass('blink');
                    } else {
                        $('#status-' + file.id).removeClass('blink');
                    }

                    if (file.status == 'Pending' || file.status == 'Chunking') {
                        pol_again = true;
                        pol_count++;
                    }
                }

                if (pol_again) {
                    $('#page-message').html(pol_count + ' File' + (pol_count == 1 ? ' is':'s are') +' being processed in the background.');
                    $('#flash-message-container').show();
                    setTimeout(pol_status, 2000);
                } else {
                    status_polling = false;
                    $('#flash-message-container').hide();
                }
            }
        },
        error: function(e) {
            chat_dialog('Error', 'Error in server-connection', ['Ok', function() {}]);
        }
    });
}

function page_ready() {
    if (parseInt($('#id').val()) >= 1) {
        var dz = $('#dropzone').dropzone({ 
            url: '/docset-upload-file/' + $('#id').val(),
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
                        // $(file.previewElement).find('.dz-progress').css('opacity', 0);
                        // $(file.previewElement).find('.dz-success-mark').css({'opacity': 1, 'margin': '22px 0 0 -42px', 'font-size': '20px', 'color': 'black'}).addClass('blink').html('Chunking');
                        if (!status_polling) {
                            pol_status();
                        }
                    }
                });
                that.on('success', function(file, response) {
                    // remove visible file/element in dropzone container
                    that.removeFile(file);
                    if (that.files.length == 0) { // All files are uploaded
                        /*
                        chat_dialog('Upload', 'Upload completed.',['View chunks', function() {
                                url = location.origin + '/docset-files/' + $('#id').val();
                                location.assign(url);
                            }], ['Ok', function(){
                                location.reload();
                            }]);
                        */
                    }
                });
                that.on('error', function(file, response) {
                    // remove visible file/element in dropzone container
                    that.removeFile(file);
                    chat_dialog('Upload', 'Error during upload.',['Ok', function() {}]);
                });
            },
        }).addClass('dropzone');
        pol_status();
    }
}

function docset_delete_file(file_id) {
    msg='Delete file \'' + $('#file_name_' + file_id).html() + '\'?';
    url = location.origin + '/docset-delete-file/' + file_id;
    chat_delete(msg, url);
}
