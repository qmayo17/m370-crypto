import math
import random

KEY_SIZE=50

def is_prime(n):
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    d = 3
    while (d*d <= n):
        if n % d == 0:
            return False
        d += 2
    return True

def generate_nbit_prime(n):
    p = random.randrange(2**(n-1) + 1, 2**n - 1)
    while (p % 2 == 0):
        p = random.randrange(2**(n-1) + 1, 2**n - 1)
    while (not is_prime(p)):
        p += 2
    return p

def generate_keys():
    p, q = generate_nbit_prime(KEY_SIZE), generate_nbit_prime(KEY_SIZE)
    n = p * q
    phi = (p-1) * (q-1)

    #Compute a unit e in Z/phiZ.
    e = random.randrange(1, phi)
    while (math.gcd(e, phi) != 1):
        e = random.randrange(1, phi)
    d = pow(e, -1, phi)

    #Return (private key, public key).
    return ((d,n), (e,n)) 

def encrypt(m, k):
    e, n = k
    if m > n:
        raise ValueError("Plaintext must be less than modulus.")
    c = pow(m, e, n)
    return c

def decrypt(c, k):
    d, n = k
    if c > n:
        raise ValueError("Ciphertext must be less than modulus.")
    m = pow(c, d, n)
    return m

if __name__ == '__main__':
    print("RSA ENCRYPTION TOOL--MATH370 FINAL PROJECT\n")
    if input("Do you already have encryption keys? (Y/N) ") in ["Y", "y"]:
        if input("Would you like to encrypt? (Y/N) ") in ["Y", "y"]:
            e, n = map(int, input("Enter your public key and RSA modulus, separated by a space: ").split(" "))
            m = int(input("Enter a message between 1 and {n}: "))
            print("Encrypting message...")
            print(f"Your ciphertext is: {encrypt(m, (e,n))}")
        elif input("Would you like to decrypt? (Y/N) ") in ["Y", "y"]:
            d, n = map(int, input("Enter your private key and RSA modulus, separated by a space: ").split(" "))
            m = int(input("Enter your ciphertext: "))
            print("Decrypting message...")
            print(f"Your message is: {decrypt(m, (d,n))}")
    else:
        privk, pubk = generate_keys()
        e = pubk[0]
        d = privk[0]
        n = pubk[1]
        print("Generating RSA parameters...")
        print(f"RSA MODULUS: {n}")
        print(f"PUBLIC KEY: {e}")
        print(f"PRIVATE KEY: {d}")
