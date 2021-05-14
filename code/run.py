import <app_pkg_name>
from <app_pkg_name>.bin import zlogger 

app = <app_pkg_name>.init_app() 

zlogger.startLogger("<app_pkg_name>")

if __name__ == "__main__":
    '''
    TODO: populate dummy data, setup zlogger 
    ''' 
    zlogger.log( "run.py" f"starting {__name__}" ) 

    app.run(debug=True)