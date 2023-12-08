function page_ready() {
    if ($('#questions').length == 1) { // The Chat page, when chatting has started...

        // Set docset_id to readonly; It may not be changed
        $('#docset_id').attr('disabled','disabled');
        // Eanable it on submit, else flask thinks docset_id is None
        $('form').on('submit', function() {
            $('[disabled=disabled]').removeAttr('disabled');
        });

        // Enable immediate typing and working of 'Enter' key
        $('#question').on('keypress', function(e) {
            if (e.keyCode == 13) {
                new_question();
            }
        }).focus();

        // Enable working of clicking on answers (show info)
        $('.answer').on('click', show_source_info);
        $('.answer span').on('click', show_source_info);


        // Set elements to window height, scroll downwards
        set_chat_height();
        document.getElementById("questions-footer").scrollIntoView(false);

    }
}

function highlight_source() {
    var url, i, sources = $('#questions-info').find('.source-item-attr-value-source');
    if (sources.length > 0) {
        url = location.origin + '/docset-chunks/' + $('#docset_id').val() + '?src=';
        for (i = 0; i < sources.length; i++) {
            url += (i == 0 ? '' : ',') + $(sources[i]).html();
        }
        window.open(url, 'chatnmdc-chunks');
    }
}


function show_panels(width_1) {
    var i, panels = $('.flex'), children, active;
    switch (width_1) {
        case 100: active = 0; break;
        case 70: active = 1; break;
        case 50: active = 2; break;
        case 30: active = 3; break;
        case 0: active = 4; break;
        default: console.log('Panel width ' + width_1 + ' is not implemented.');
    }
    for (i = 0; i < panels.length; i++) {
        children = $(panels[i]).children();
        children.removeClass('flex0 flex30 flex50 flex70 flex100');
        $(children[0]).addClass('flex' + width_1);
        $(children[1]).addClass('flex' + (100 - width_1));
    }
    children = $('.questions-container-header').children();
    children.removeClass('active');
    $(children[active]).addClass('active');
}

function set_chat_height() {
    var pos = $('#questions').position();
    $('#questions').css('max-height', (window.innerHeight - pos.top) + 'px');
    $('#questions-info').css('max-height', (window.innerHeight - pos.top) + 'px');
}

function toggle_chat_form_header() {
    $('.chat-form').toggle();
    $('#chat-form-title').toggle();
    set_chat_height();
}

function show_source_info(e) {
    var html='', i, j, source, key, value, keys, sources, el=$(e.target);
    if (el.prop('tagName') == 'SPAN') {el = $(el.parent());}
    sources = el.parent().find('.source').html();
    if (sources != '') {
        sources = JSON.parse(sources);
        for (i in sources) {
            html += '<div class="source-item-header">Source item #' + (i + 1) + '</div>';
            source = sources[i];
            keys = Object.keys(source);
            html += '<div class="source-item alert alert-info">';
            for (j in keys) {
                key = keys[j];
                value = source[key];
                key = key[0].toUpperCase() + key.substring(1);
                html += '<div class="source-item-attr">';
                html += '<div class="source-item-attr-key">' + key.replaceAll('_',' ') + ':</div>';
                html += '<div class="source-item-attr-value' + (key == 'Source' ? ' source-item-attr-value-source': '') + '">' + value + '</div>';
                html += '</div>';
            }
            html += '</div>';
        }
        $('#a-highlight-source').removeClass('disabled');
    } else {
        html = '<div class="questions-no-info alert alert-danger">No sources found.</div>';
        $('#a-highlight-source').addClass('disabled');
    }
    if (html == '') {
        html = '<div class="questions-no-info alert alert-warning">The AI-model didn\'t use any of the documents in \'' + $('#chat-form-title').find('h2').html() + '\'.</div>';
        $('#a-highlight-source').addClass('disabled');
    }
    $('#questions-info').html(html);
}

function chat_model_info() {
    chat_dialog('Model parameters', $('#chat-info-table').html(), ['Ok', function(){}]);
}


function new_question() {
    var url = window.location.origin + '/question/' + $('#id').val(), question=$('#question').val().trim();
    if (question != '') {
        $('new-question-button').hide();
        $('#chat_progress_spinner').removeClass('chat-spinner-done');
        $.ajax({
            type: "POST",
            url: url,
            timeout: 0,
            data: {'question': question},
            success: function (data) {
                data = data.data;
                if (data.error) {
                    chat_dialog('Error', data.msg, ['Ok', function() {window.location.reload();}]);
                } else {
                    window.location.reload();
                }
            },
            error: function(e) {
                chat_dialog('Error', 'Error in server-connection', ['Ok', function() {}]);
            }
        });
        new_question_progress(0);
    }
}

function new_question_progress(t) {
    var txt;
    if (t < 60) {
        txt = t + ' seconds';
    } else if (t < 3600) {
        txt = new Date(t * 1000).toISOString().substring(14, 19)
    } else {
        txt = new Date(t * 1000).toISOString().substring(11, 16);
    }
    $('#chat_progress_').html('Working: ' + txt);    
    setTimeout(new_question_progress, 1000, t+1);
}
