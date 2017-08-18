"""UncertaintyWrapper"""

__VERSION__ = '0.4.2a1'
__RELEASE__ = u"Terreneuvian Series"
__URL__ = u'https://github.com/SunPower/UncertaintyWrapper'
__AUTHOR__ = u"Mark Mikofski"
__EMAIL__ = u'mark.mikofski@sunpowercorp.com'

try:
    from uncertainty_wrapper.core import unc_wrapper, unc_wrapper_args, jflatten
except ImportError:
    pass
else:
    __all__ = ['unc_wrapper', 'unc_wrapper_args', 'jflatten']
