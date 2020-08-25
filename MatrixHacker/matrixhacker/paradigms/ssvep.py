from .base import BaseParadigm


class SSVEP(BaseParadigm):
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != 'ssvep':
            ret = False

        if self.event_list:
            if not set(self.event_list) <= set(dataset.event_info.keys()):
                ret = False

        # case insensitive strict channel matching check
        if self.select_channels:
            if not set(self.select_channels) <= set(dataset.channels):
                ret = False
                
        return ret
        