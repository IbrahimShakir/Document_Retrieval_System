from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Document(db.Model):
    __tablename__ = 'documents'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    vector = db.Column(db.PickleType, nullable=False)  # Store vector as pickled object

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(50), primary_key=True)
    request_count = db.Column(db.Integer, default=0)
