from fto.detection.atp import RESTARTABLE_HINTS

class RestartConfig:
    """Singleton allow-list of restart cases.
    """

    _instance: 'RestartConfig | None' = None

    def __init__(self, codes=None, sites=None, families=None) -> None:
        self.codes = set(codes) if codes else set()
        self.sites = set(sites) if sites else set()
        self.families = set(families) if families else set()
        self.default = not (self.codes or self.sites or self.families)

    @classmethod
    def instance(cls) -> 'RestartConfig':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def configure(cls, codes=None, sites=None, families=None) -> 'RestartConfig':
        cls._instance = cls(codes, sites, families)
        return cls._instance

    @property
    def configured(self) -> bool:
        return bool(self.codes or self.sites or self.families)

    def restartable(self, code=None, site=None, family=None, recovery_hint=None) -> bool:
        if self.default:
            return recovery_hint in RESTARTABLE_HINTS
        return (
            (self.codes and code in self.codes)
            or (self.sites and site in self.sites)
            or (self.families and family in self.families)
        )
