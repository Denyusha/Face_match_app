from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId

class User:
    def __init__(self, username, email, password, is_admin=False):
        self.username = username
        self.email = email
        self.password = generate_password_hash(password)
        self.is_admin = is_admin
        self.created_at = datetime.utcnow()

    @staticmethod
    def create_user(db, user_data):
        # Check if username or email already exists
        if db.users.find_one({'$or': [{'username': user_data['username']}, {'email': user_data['email']}]}):
            return None
        
        user = User(
            username=user_data['username'],
            email=user_data['email'],
            password=user_data['password'],
            is_admin=user_data.get('is_admin', False)
        )
        
        # Convert user object to dictionary
        user_dict = {
            'username': user.username,
            'email': user.email,
            'password': user.password,
            'is_admin': user.is_admin,
            'created_at': user.created_at
        }
        
        # Insert into database
        result = db.users.insert_one(user_dict)
        user_dict['_id'] = result.inserted_id
        return user_dict

    @staticmethod
    def get_user_by_username(db, username):
        user = db.users.find_one({'username': username})
        return user

    @staticmethod
    def verify_password(stored_password, provided_password):
        return check_password_hash(stored_password, provided_password) 