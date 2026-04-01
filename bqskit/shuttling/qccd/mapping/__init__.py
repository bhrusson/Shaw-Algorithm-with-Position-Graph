from .layout.pam import QCCDPAMLayoutPass
from .layout.naive import QCCDLayoutPass
from .routing.pam import QCCDPAMRoutingPass
from .routing.naive import QCCDRoutingPass

try:
    from .layout.pam_pgs import QCCDPAMLayoutPassPGS
    from .layout.naive_pgs import QCCDLayoutPassPGS
    from .routing.pam_pgs import QCCDPAMRoutingPassPGS
    from .routing.naive_pgs import QCCDRoutingPassPGS
except ModuleNotFoundError:
    # Allow legacy non-PGS workflows to import this package even when the
    # optional PGS dependency chain is unavailable in the current execution
    # environment.
    pass
