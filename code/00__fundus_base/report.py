'''
author: bg
goal: 
type: Logger and Reports worker 
how: 
ref: 
refactors: 
'''

from datetime import datetime

## TODO: refactor @ worker, design pattern, logger module, fileIO, colored, ZPdDataStats
class ZReporter:
    worker = None 
    LOG_ENTRY = "{:18s}: {:s}".format
    LOG = True 
    LOG_DIR = "."

    class ZWorker():        
        def __init__(self, name):
            self.name = name
            self.queue = []
            self.running = False 

        def start(self):
            self.running = True 

        def run(self):
            if self.running: ## TODO: new thread + while running 
                for rec in self.queue:
                    self._log_to_file(rec) 

        def _log_to_file(self, rec):
            with open(f"{ZReporter.LOG_DIR}/{self.name}.txt", 'a') as fd:
                fd.writelines( f"{rec}\n" ) 


    
    @staticmethod
    def start(name=None, log_screen=True):  
        ZReporter.LOG = log_screen 
        if ZReporter.worker is None:
            ZReporter.worker = ZReporter.ZWorker(name) 
        ZReporter.worker.start()
    
    @staticmethod
    def add_log_entry(rec):        
        if ZReporter.worker is None:
            ZReporter.start('ZTEST') 

        d = datetime.today().strftime("%d-%m-%Y %H:%M:%S")
        
        outiez = ZReporter.LOG_ENTRY(d, rec) 
        ZReporter.worker.queue.append( outiez )
        ZReporter.worker.run() ## TODO: at multithreading/tasking 

        if ZReporter.LOG:
            print( outiez )  


if __name__ == "__main__":
    ZReporter.add_log_entry("We're heading for the stars and the moon is enroute!")
    
