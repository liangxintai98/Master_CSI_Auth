import socket
from datetime import datetime
import secure_connection
from cryptography.hazmat.backends import default_backend  
from cryptography.hazmat.primitives.asymmetric import padding  
from cryptography.hazmat.primitives import hashes  
from cryptography.hazmat.primitives.serialization import load_pem_private_key  
from cryptography.hazmat.primitives.serialization import load_pem_public_key  
import time
import os
from os.path import exists
import tracemalloc
import psutil
#from Crypto.Cipher import AES
#from Crypto.Util.Padding import pad, unpad
from binascii import unhexlify

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():

    p = psutil.Process(os.getpid())
    cpu_rec = []

    #Get the server's public key and clients private key for authentication (Signeture) and sending session keys
    serverPubKey = load_pem_public_key(open('rsapub.pem', 'rb').read(),default_backend()) 
    clientPvKey = load_pem_private_key(open('rsapv_client_2048.pem', 'rb').read(),b"12345",default_backend())
    iv = "7bde5a0f3f39fd658efc45de143cbc94"
    iv = unhexlify(iv)

    #Here we define some global variables for handling session keys time expiration 


    # HOST = '127.0.0.1'  # The server's hostname or IP address
    HOST = '192.168.3.1'
    PORT = 8889     # The port used by the server
    expiration_time = 20  #Expiration time for session key in >>Seconds<<
    auth_method = 'digital'
    # auth_method = 'hamc'

    # print(ciphertext)
    # alicePrivKey = load_pem_private_key(open('rsakey.pem', 'rb').read(),b"12345",default_backend())  

    # d = alicePrivKey.decrypt(ciphertext,  
    #                                 padding.OAEP(  
    #             mgf=padding.MGF1(algorithm=hashes.SHA256()),  
    #             algorithm=hashes.SHA256(),  
    #             label=None  
    #             )  
    # ) 
    # print(d)
    if __name__ == "__main__":
        session_key = b""
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # s.connect((HOST, PORT))
            # if secure_connection.key_is_expired(expiration_time):
            # s.sendall(b"NEW_KEY")
            s.sendto(b"NEW_KEY", (HOST,PORT))
            session_key = secure_connection.generate_session_key()
            # cipher = AES.new(session_key, AES.MODE_CBC, iv)
            # print(session_key.decode('utf-8'))
            ciphered_session_key = secure_connection.encrypt_session_key(session_key, serverPubKey)
            # print(ciphered_session_key)
            # s.sendall(b"NEW_KEY")
            # s.sendall(ciphered_session_key)
            s.sendto(ciphered_session_key, (HOST,PORT))

            # result = b'0'
            n = 0
            time.sleep(2)

            while True:

                    # if secure_connection.key_is_expired(expiration_time):
                    #     session_key = secure_connection.generate_session_key()
                    #     ciphered_session_key = secure_connection.encrypt_session_key(session_key, serverPubKey)
                    #     s.sendall(b"NEW_KEY")
                    #     s.sendall(ciphered_session_key)
                    #Here we tell the server tha we are sending a text  
                start = time.time()
                # file_exist = exists('/Users/liangxintai/Desktop/live_pcap/client_profile.pcap')
                # if file_exist == False:
                #     print('User profile building ...')
                    
                print("Message sending ...")
                text = 'Hello World!'
                # s.sendall(b"TEXT")
                text = text.encode()
                # text = pad(text, AES.block_size)
                ciphered_text = secure_connection.encrypt_message(text, session_key)
                # ciphered_text = cipher.encrypt(text)
                # print(len(ciphered_text))
                # s.sendall(ciphered_text)
                
                # wether User_profile exist
                user_profile = b'0'

                # if user_profile == b'0':
                #     print("User_profile building ...")
                #     result = b'0'

                # elif user_profile == b'1':

                result = s.recv(1024)

                if result == b'1':
                    signature = b'no_signature'
                elif result == b'0':
                    if auth_method == 'digital':
                        signature = secure_connection.sign_message(text, clientPvKey)
                    if auth_method == 'hamc':
                        signature = secure_connection.hmac_gen(text, session_key).encode()
                        

                # s.sendall(b'$$$$$'.join([b"TEXT", ciphered_text, signature]))
                s.sendto(b'$$$$$'.join([b"TEXT", ciphered_text, signature]),(HOST,PORT))

                ciphered_text_2, signature_2 = [i for i in s.recv(1024).split(b'$$$$$')]
                plain_text = secure_connection.decrypt_message(ciphered_text_2, session_key)
                print(plain_text.decode())  # print response from server

                if result == b'1':
                    print("SIGNATURE check NO NEED")
                elif result == b'0':
                    print("CSI authentication FAIL: checking SIGNATURE ...")
                    if secure_connection.signature_is_valid(plain_text, signature_2, serverPubKey):
                        print("SIGNATURE authentication SUCCESS")
                    else:
                        print("SIGNATURE authentication FAIL")

                # current, peak = tracemalloc.get_traced_memory()
                # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                # result = s.recv(1024)
                
                if n%10 == 0:
                    us = p.cpu_percent()
                    # print('*****************')
                    print(f"current CPU usage is {us}\n")
                    # print('*****************\n')

                    cpu_rec.append(us)
		 
                    f = open("cpu_client_time.txt", "w")
                
                    for d in cpu_rec:
                        f.write(f"{d}\n")
                    f.close()

                end = time.time()
                # print(end - start)
                n = n + 1

                time.sleep(0.05)

if __name__ == "__main__":

    # tracemalloc.start()
    main()
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # tracemalloc.stop()_new
