function page_ready() {
    var i, srcs = location.search;
    if (srcs.substring(0, 5) == '?src=') {
        srcs = srcs.substring(5).split(',');
    } else {
        srcs = [];
        $('#unused-chunks').hide();
    }
    for (i = 0; i < srcs.length; i++) {
        $('[source=' + srcs[i] + ']').addClass('chunk-highlight').click();
    }

    // Set vars for 'init search' and 'find overlap'
    var chunks=$('.chunk'), l = chunks.length, i, chunk_parts = [], chunk_parts_txt = [];
    for (i = 0; i < chunks.length; i++) {
        tmp = $(chunks[i]).children();
        chunk_parts[i] = [$(tmp[0]), $(tmp[1]), $(tmp[2])];
        chunk_parts_txt[i] = [chunk_parts[i][0].html(), chunk_parts[i][1].html(), chunk_parts[i][2].html()];
    }

    // init search
    $('#chunk-search').on('keyup', function(e) {
        var txt, search = $('#chunk-search').val().toLowerCase();

        e.stopPropagation();
        for (i = 0; i < l; i++) {
            txt = (chunk_parts_txt[i][0] + chunk_parts_txt[i][1] + chunk_parts_txt[i][2]).toLowerCase();
            if (txt.indexOf(search) >= 0) {
                $(chunks[i]).parent().removeClass('chunk-hide-all');
            } else {
                $(chunks[i]).parent().addClass('chunk-hide-all');
            }
        }
        return false;
    });

    // find overlap
    var j, prev_chunk_txt, this_chunk_txt
    for (i = 1; i < l; i++) { // start at 1: Find overlap between i - 1 and i
        j = 1;
        prev_chunk_txt = chunk_parts_txt[i - 1][0] + chunk_parts_txt[i - 1][1] + chunk_parts_txt[i - 1][2];
        this_chunk_txt = chunk_parts_txt[i][0] + chunk_parts_txt[i][1] + chunk_parts_txt[i][2];
        while (j < prev_chunk_txt.length && j < this_chunk_txt.length) {
            if (prev_chunk_txt.slice(-j) == this_chunk_txt.substring(0, j)) {
                // There is an overlap of j characters

                // Handle the previous chunk
                if (j <= chunk_parts_txt[i - 1][2].length) { // Overlap within right span
                    chunk_parts[i - 1][2].html(chunk_parts_txt[i - 1][2].substring(0, chunk_parts_txt[i - 1][2].length - j) + '<span class="chunk-overlap-end">' + chunk_parts_txt[i - 1][2].slice(-j) + '</span>');
                } else if (j <= chunk_parts_txt[i - 1][1].length + chunk_parts_txt[i - 1][2].length) { // overlap in middle and right span
                    jj = j - chunk_parts_txt[i - 1][2].length;
                    chunk_parts[i - 1][1].html(chunk_parts_txt[i - 1][1].substring(0, chunk_parts_txt[i - 1][1].length - jj) + '<span class="chunk-overlap-end">' + chunk_parts_txt[i - 1][1].slice(-jj) + '</span>');
                    chunk_parts[i - 1][2].html('<span class="chunk-overlap-end">' + chunk_parts_txt[i - 1][2] + '</span>');
                } else { // Overlap in all spans
                    jj = j - chunk_parts_txt[i - 1][1].length - chunk_parts_txt[i - 1][2].length;
                    chunk_parts[i - 1][0].html(chunk_parts_txt[i - 1][0].substring(0, chunk_parts_txt[i - 1][0].length - jj) + '<span class="chunk-overlap-end">' + chunk_parts_txt[i - 1][0].slice(-jj) + '</span>');
                    chunk_parts[i - 1][1].html('<span class="chunk-overlap-end">' + chunk_parts_txt[i - 1][1] + '</span>');
                    chunk_parts[i - 1][2].html('<span class="chunk-overlap-end">' + chunk_parts_txt[i - 1][2] + '</span>');
                }

                // handle this chunk
                if (j <= chunk_parts_txt[i][0].length) { // Overlap within left span
                    chunk_parts[i][0].html('<span class="chunk-overlap-start chunk-overlap-show">' + chunk_parts_txt[i][0].substring(0, j) + '</span>' + chunk_parts_txt[i][0].slice(-j));
                } else if (j <= chunk_parts_txt[i][0].length + chunk_parts_txt[i][1].length) { // overlap in left and middle span
                    jj = j - chunk_parts_txt[i][0].length;
                    chunk_parts[i][0].html('<span class="chunk-overlap-start chunk-overlap-show">' + chunk_parts_txt[i][0] + '</span>');
                    chunk_parts[i][1].html('<span class="chunk-overlap-start chunk-overlap-show">' + chunk_parts_txt[i][1].substring(0, jj) + '</span>' + chunk_parts_txt[i][1].slice(-jj));
                } else { // Overlap in all spans
                    jj = j - chunk_parts_txt[i][0].length - chunk_parts_txt[i][1].length;
                    chunk_parts[i][0].html('<span class="chunk-overlap-start chunk-overlap-show">' + chunk_parts_txt[i][0] + '</span>');
                    chunk_parts[i][1].html('<span class="chunk-overlap-start chunk-overlap-show">' + chunk_parts_txt[i][1] + '</span>');
                    chunk_parts[i][2].html('<span class="chunk-overlap-start chunk-overlap-show">' + chunk_parts_txt[i][2].substring(0, jj) + '</span>' + chunk_parts_txt[i][2].slice(-jj));
                }
                break;
            }
            j++;
        }
    }
}

function chunk_search_clear() {
    $('#chunk-search').val('');
    $('.chunk-outer').removeClass('chunk-hide-all');
}

function unused_chunks() {
    $('.chunk-outer:not(.chunk-highlight)').toggle();
    var el = $('#unused-chunks-hide');
    if (el.attr('chunk_hide') == 'hide') {
        el.attr('chunk_hide','show').html('Visible');
    } else {
        el.attr('chunk_hide','hide').html('Hidden');
    }    
}

function middle_sections() {
    var el = $('#middle-sections-hide');
    if (el.attr('middle_hide') == 'show') {
        $('.chunk-outer').addClass('chunk-hide');
        el.attr('middle_hide','hide').html('Hidden');
    } else {
        $('.chunk-outer').removeClass('chunk-hide');
        el.attr('middle_hide','show').html('Visible');
    }    
}

function show_overlap() {
    var el = $('#show_overlap'), show = el.attr('show_overlap'), first=false, last=false, txt;
    switch (show) {
        case 'start': txt='End';   show='end';   last=true;             break;
        case 'end':   txt='Both';  show='both';  first=true; last=true; break;
        case 'both':  txt='Start'; show='start'; first=true;            break;
    }
    el.html(txt).attr('show_overlap', show);
    if (first) {
        $('.chunk-overlap-start').addClass('chunk-overlap-show');
    } else {
        $('.chunk-overlap-start').removeClass('chunk-overlap-show');
    }
    if (last) {
        $('.chunk-overlap-end').addClass('chunk-overlap-show');
    } else {
        $('.chunk-overlap-end').removeClass('chunk-overlap-show');
    }
}
