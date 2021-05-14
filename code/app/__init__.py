from flask import Flask
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy 

from  <app_pkg_name>.config import Config

from <app_pkg_name>.bin import zlogger 

## 1. load config from json file or environ 
conf = Config()

## 2. login manager
login_manager = LoginManager() 
login_manager.login_view = 'users.login' ##indicate login route 
login_manager.login_message_category = 'info' 

## 3. hashing service 
bcrypt = Bcrypt()

## 4. sql db 
db = SQLAlchemy() 


## 0. app factory method 
def init_app(config_obj=conf): 
    zlogger.log( "app.init_app", f"{config_obj}" ) 
    
    zlogger.log( "app_pkg.py", f": {__name__}" )
    
    app = Flask(__name__) 
    app.config.from_object( config_obj ) 

    db.init_app( app )
    login_manager.init_app( app ) 
    bcrypt.init_app( app ) 

    from <app_pkg_name>.errors.handlers import errors 
    from <app_pkg_name>.faq.routes import faqs       

    app.register_blueprint( errors ) 
    app.register_blueprint( faqs )  
    
    ## TODO: disable if not in use
    with app.app_context():
        db.create_all()
        db.session.commit()

    return app 