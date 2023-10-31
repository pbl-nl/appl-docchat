function chat_dialog(title, message, button1, button2) {
    var dialog=$('#chat-dialog');
    dialog.find('.modal-title').html(title);
    dialog.find('.modal-body').html('<p>' + message + '</p>');
    dialog.find('.modal-header').find('.close').show().unbind('click').click(function() {dialog.modal('hide');});
    if (button1) {
        dialog.find('.modal-footer').find('.btn-primary').html(button1[0]).show().unbind('click').click(function() {button1[1](); dialog.modal('hide');});
    } else {
        dialog.find('.modal-footer').find('.btn-primary').hide();
        dialog.find('.modal-header').find('.close').hide();
    }
    if (button2) {
        dialog.find('.modal-footer').find('.btn-secondary').html(button2[0]).show().unbind('click').click(function() {button2[1](); dialog.modal('hide');});
    } else {
        dialog.find('.modal-footer').find('.btn-secondary').hide();
    }
    dialog.modal('show');
}

function chat_delete(msg, url) {
    chat_dialog('Delete', msg, ['Yes', function() {location.assign(url);}], ['No', function() {dialog.modal('hide');}]);
}

var docCookies = {
  getItem: function (sKey) {
    return decodeURIComponent(document.cookie.replace(new RegExp("(?:(?:^|.*;)\\s*" + encodeURIComponent(sKey).replace(/[\-\.\+\*]/g, "\\$&") + "\\s*\\=\\s*([^;]*).*$)|^.*$"), "$1")) || null;
  },
  setItem: function (sKey, sValue, vEnd, sPath, sDomain, bSecure) {
    if (!sKey || /^(?:expires|max\-age|path|domain|secure)$/i.test(sKey)) { return false; }
    var sExpires = "";
    if (vEnd) {
      switch (vEnd.constructor) {
        case Number:
          sExpires = vEnd === Infinity ? "; expires=Fri, 31 Dec 9999 23:59:59 GMT" : "; max-age=" + vEnd;
          break;
        case String:
          sExpires = "; expires=" + vEnd;
          break;
        case Date:
          sExpires = "; expires=" + vEnd.toUTCString();
          break;
      }
    }
    document.cookie = encodeURIComponent(sKey) + "=" + encodeURIComponent(sValue) + sExpires + (sDomain ? "; domain=" + sDomain : "") + (sPath ? "; path=" + sPath : "") + (bSecure ? "; secure" : "");
    return true;
  },
  removeItem: function (sKey, sPath, sDomain) {
    if (!sKey || !this.hasItem(sKey)) { return false; }
    document.cookie = encodeURIComponent(sKey) + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT" + ( sDomain ? "; domain=" + sDomain : "") + ( sPath ? "; path=" + sPath : "");
    return true;
  },
  hasItem: function (sKey) {
    return (new RegExp("(?:^|;\\s*)" + encodeURIComponent(sKey).replace(/[\-\.\+\*]/g, "\\$&") + "\\s*\\=")).test(document.cookie);
  },
  keys: /* optional method: you can safely remove it! */ function () {
    var aKeys = document.cookie.replace(/((?:^|\s*;)[^\=]+)(?=;|$)|^\s*|\s*(?:\=[^;]*)?(?:\1|$)/g, "").split(/\s*(?:\=[^;]*)?;\s*/);
    for (var nIdx = 0; nIdx < aKeys.length; nIdx++) { aKeys[nIdx] = decodeURIComponent(aKeys[nIdx]); }
    return aKeys;
  }
};



function javascript_ready_general() {
    // Tab's
    // Bootstrap dropdown's
    $('.navbar-toggler').on('click', function(e) {
        $('#navbarSupportedContent').toggleClass('show');
    });
    var dropdownElementList = [].slice.call(document.querySelectorAll('.dropdown-toggle'));
    var dropdownList = dropdownElementList.map(function (dropdownToggleEl) {
        return new bootstrap.Dropdown(dropdownToggleEl);
    });
    
    $('.table-sortable').DataTable({});
    try {
        page_ready();
    } catch(e) {}
}