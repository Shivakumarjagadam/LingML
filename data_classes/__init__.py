from .constraint_covid import ConstraintCovid
from .covid_mis_20 import CovidMis20
from .constraint_covid_cleaned import ConstraintCleaned
from .constraint_covid_appended import ConstraintAppended
from .constraint_covid_cleaned_appended import ConstraintCleanedAppended


def dataclass_map(key):
    
    if key == 'aaai-constraint-covid':
        return ConstraintCovid
    if key == 'aaai-constraint-covid-cleaned': 
        return ConstraintCleaned
    if key == 'aaai-constraint-covid-appended': 
        return ConstraintAppended
    if key == 'aaai-constraint-covid-cleaned-appended': 
        return ConstraintCleanedAppended
    if key == 'covid-misinformation': 
        return CovidMis20