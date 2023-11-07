function page_ready() {
    if (parseInt($('#id').val()) >= 1) {
        $('form').find('input,select').each(function(i, el) {
            var id = $(el).prop('id');
            if (id != 'id' && id != 'csrf_token') {
                $(el).attr('disabled', 'disabled');
            }
        });
        $('#name,.btn-submit').removeAttr('disabled');
    }
}