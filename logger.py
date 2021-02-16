from abc import ABC, abstractmethod
import os
import csv

class Logger(ABC): 
    def __init__(self, logpath): 
        self.logpath = logpath
        
    @abstractmethod
    def log(self, list): 
        pass
    
class CSVLogger(Logger): 
    def __init__(self, logpath, rows): 
        super().__init__(logpath)
        self.rows = rows
        
        if not os.path.exists(logpath):
            with open(logpath, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(rows)

    def log(self, list): 
        with open(self.logpath, 'a') as logfile: 
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(list)
        