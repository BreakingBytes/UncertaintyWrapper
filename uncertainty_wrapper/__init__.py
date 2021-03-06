"""UncertaintyWrapper"""

__VERSION__ = '0.4.2'
__RELEASE__ = u"Terreneuvian Series"
__URL__ = u'https://github.com/BreakingBytes/UncertaintyWrapper'
__AUTHOR__ = u"Mark Mikofski"
__EMAIL__ = u'bwana.marko@yahoo.com'

try:
    from uncertainty_wrapper.core import unc_wrapper, unc_wrapper_args, jflatten
except (ImportError, ModuleNotFoundError):
    pass
else:
    __all__ = ['unc_wrapper', 'unc_wrapper_args', 'jflatten']
