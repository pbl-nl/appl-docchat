function page_ready() {
    // remove margin-bottom to 'lift image up'
    $('#menu-container').removeClass('mb-5');
    // adjust margin in case of alerts
    if ($('#flash-message-container').length >= 1) {
        $('#flash-message-container').css('margin', '4px auto -14px auto')
    }
}
