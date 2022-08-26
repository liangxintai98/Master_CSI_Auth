from pickletools import dis
from cryptography.fernet import Fernet 
import time
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend  
from cryptography.hazmat.primitives.asymmetric import padding  
from cryptography.hazmat.primitives import hashes 
import hmac
 
'''

private_key = rsa.generate_private_key(public_exponent=65537,key_size=4096,backend=default_backend())
public_key = private_key.public_key()

# Save the private key in PEM format encrypted
with open("rsapv_client_4096.pem", "wb") as f:  
    f.write(private_key.private_bytes(  
        encoding=serialization.Encoding.PEM,  
        format=serialization.PrivateFormat.TraditionalOpenSSL,  
        encryption_algorithm=serialization.BestAvailableEncryption(b"12345"),  
    )  
)  
  
# Save the Public key in PEM format  
with open("rsapub_client_4096.pem", "wb") as f:  
    f.write(public_key.public_bytes(  
        encoding=serialization.Encoding.PEM,  
        format=serialization.PublicFormat.SubjectPublicKeyInfo,  
    )  
)

'''


def save_time(time):
    f = open("time.txt", "w+")
    f.seek(0)
    f.write(str(time))
    f.close

def load_time():
    f = open("time.txt", "r")
    current_time = float(f.read())
    print(current_time)
    return current_time

def check_time():
    ts = time.time()
    return ts 


def key_is_expired(expiration_time):
    now = check_time()
    old = load_time()
    if now - old > expiration_time:
        print("key is expired")
        return True
    else:
        print("key is valid yet")
        return False

def generate_session_key():
    key = Fernet.generate_key()
    ts = time.time()
    print("Session key generated!")
    older_key_time = load_time()
    # save_time(ts)
    return key

def encrypt_session_key(session_key, public_key):
    ciphered_session_key = public_key.encrypt(  
    session_key,  
    padding.OAEP(  
            mgf=padding.MGF1(algorithm=hashes.SHA256()),  
            algorithm=hashes.SHA256(),  
            label=None  
        )  
    ) 
    return ciphered_session_key

def decrypt_session_key(ciphered_session_key, private_key):
    session_key = private_key.decrypt(  
    ciphered_session_key,  
    padding.OAEP(  
            mgf=padding.MGF1(algorithm=hashes.SHA256()),  
            algorithm=hashes.SHA256(),  
            label=None  
        )  
    )  
    return session_key 


def encrypt_message(plaintext, session_key):
    e = Fernet(session_key)
    ciphertext = e.encrypt(plaintext)
    return ciphertext

def decrypt_message(ciphertext, session_key):
    d = Fernet(session_key)
    plaintext = d.decrypt(ciphertext)
    return plaintext

def sign_message(message, private_key):
    sig = private_key.sign(  
    message,  
    padding.PSS(  
        mgf=padding.MGF1(algorithm=hashes.SHA256()),  
        salt_length=padding.PSS.MAX_LENGTH,  
    ),  
    hashes.SHA256()  
    )
    return sig

def signature_is_valid(message, signature, public_key): 
    try:
        validation = public_key.verify(  
            signature,  
            message,  
            padding.PSS(  
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),  
                    salt_length=padding.PSS.MAX_LENGTH,  
            ),  
            hashes.SHA256()  
        ) 
    except :
        print("signature verified FAIL")
        return False
    else : 
        print("signaure verified SUCCESS")
        return True

def hmac_gen(messgae, seesion_key):

    sig = hmac.new(seesion_key, messgae, digestmod= 'sha256')

    return sig.hexdigest()


# 



# print(my_key, time, older_time)




