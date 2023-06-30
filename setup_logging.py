"""
This file contains misc functions to setup the logging system.
"""

import logging

# Filter, accessed in yaml logger setup file
class infoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO
