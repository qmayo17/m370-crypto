import random

BIG_BYTES = 24
USING_RSA = False

def gcd(a, b):
  if (b == 0):
    return a
  if (a < b):
    return gcd(b, a)
  return gcd(b, a % b)

def euclid(a, b):
  if (b == 0):
    return (1, 0)
  q,r = divmod(a,b)
  x0,y0 = euclid(b,r)
  new_x = y0
  new_y = x0 - (q*y0)
  return (new_x, new_y)

def inverse(a,n):
  x,y = euclid(a,n)
  return x % n

def divide(a,b,n):
  return (a * inverse(b,n)) % n

def power(a,k,n):
  if (k==0):
    return 1
  prev = power(a, k>>1, n)
  ret = (prev*prev) % n
  if ((k & 1) == 1):
    return (ret*a) % n
  return ret

def largeInt(n):
  n2 = n
  ret = 0
  while (n2>0):
    ret = ret << 1
    if (random.random() < 0.5):
      ret += 1
    n2 = n2 >> 1
  if (ret==0 or ret >= n):
    return largeInt(n)
  return ret

def sdecomp(n):
  if (n&1 == 1):
    return (0,n)
  s0,d0 = sdecomp(n>>1)
  return (s0+1,d0)

def singleRabinMiller(n,a):
  if (n&1 == 0):
    return False
  if gcd(n,a)>1:
    return False
  n_one = n-1
  s,d = sdecomp(n_one)
  cur = power(a,d,n)
  if (cur==1 or cur==n_one):
    return True
  for nothing in range(s):
    cur = (cur*cur)%n
    if (cur == n_one):
      return True
  return False

def multiRabinMiller(n,times=100):
  for nothing in range(times):
    a = largeInt(n)
    if (not singleRabinMiller(n,a)):
      return False
  return True

def find_prime(lower = 2 << 192, upper = 2 << 193, times = 100):
  while True:
    p = largeInt(upper-lower) + lower
    if multiRabinMiller(p,times):
      return p

def isQuadraticResidue(p,n):
  #Is n a quadratic residue mod p?
  if (n==0):
    return True
  return power(n,(p-1)>>1, p) == 1

def findNonQuadRes(p):
  while True:
    z = largeInt(p)
    if not isQuadraticResidue(p,z):
      return z%p

def TonelliShanks(p,n):
  if not isQuadraticResidue(p,n):
    return False
  S,Q = sdecomp(p-1)
  z = findNonQuadRes(p)
  #print(f"Found non-quadratic residue {z}")
  M = S
  c = power(z,Q,p)
  t = power(n,Q,p)
  #R = power(n, (Q+1) << 1, p)
  #THE BITSHIFT WAS THE WRONG WAY
  R = power(n, (Q+1) >> 1, p)
  while True:
    #print(f"t is now {t}, R is now {R}")
    if (t==0):
      return 0
    if (t==1):
      return R
    least_i = 0
    t_to_2i = t
    #print(f"M is now {M}, t_to_2i is now {t_to_2i}")
    for i in range(1,M):
      t_to_2i = (t_to_2i*t_to_2i) % p
      #print(f"t_to_2i is now {t_to_2i}")
      if (t_to_2i == 1):
        least_i = i
        break
    #print(f"least_i is now {least_i}")
    b = power(c, 1 << (M-least_i-1), p)
    M = least_i
    c = (b*b)%p
    t = (t*b*b)%p
    R = (R*b)%p

#print(TonelliShanks(43,21))

#notprime = (1 << 97) - 1
#yesprime = (1 << 107) - 1
#print(multiRabinMiller(notprime,6))
#print(multiRabinMiller(yesprime,6))
#print(multiRabinMiller(yesprime*101,6))

class EllipticCurve:
  def __init__(self,n,lin_coef,con_coef):
    self.n=n
    self.lin_coef=lin_coef % n
    self.con_coef=con_coef % n
  
  def is_valid(self):
    return ((4*power(self.lin_coef,3,self.n)) + (27*power(self.con_coef,2,self.n)))%self.n != 0
  
  def isElement(self,p):
    if (p is None):
      return True
    try:
      x,y = p
      ex1 = (x*x*x) + (self.lin_coef*x) + (self.con_coef)
      ex2 = y*y
      return (ex1-ex2) % self.n == 0
    except TypeError:
      return False

  def plus(self,p,q):
    if (p is None):
      return q
    if (q is None):
      return p
    
    xp,yp = p
    xq,yq = q
    if (xp == xq and (yp+yq)%self.n == 0):
      return None
    
    slope_num = ((xp*xp) + (xp*xq) + (xq*xq) + self.lin_coef) % self.n
    slope_den = (yp+yq) % self.n
    s = divide(slope_num,slope_den,self.n)

    xr = ((s*s) - (xp+xq)) % self.n
    yr = (yp - (s*(xp - xr))) % self.n
    yr2 = (-1*yr) % self.n

    return (xr,yr2)
  
  def times(self,p,k):
    if (k==0):
      return None
    prev = self.times(p, k>>1)
    ret = self.plus(prev,prev)
    if ((k & 1) == 1):
      return self.plus(ret,p)
    return ret
  
  def findRandomElement(self):
    while True:
      x = largeInt(self.n)
      ex1 = (x*x*x) + (self.lin_coef*x) + (self.con_coef)
      ex1 = ex1 % self.n
      if isQuadraticResidue(self.n, ex1):
        y = TonelliShanks(self.n, ex1)
        return (x,y)

#z89 = EllipticCurve(89,88,0)
#p1 = (42,49)
#p2 = (42,40)
#p3 = (34,56)
#print(z89.isElement(p1),z89.isElement(p2))
#print(z89.plus(p1,p2))
#print(z89.plus(p1,p1))
#print(z89.times(p1,2))
#print(z89.plus(p1,p3))

#z61 = EllipticCurve(61,60,0)
#print(z61.is_valid())
#q1 = (24,40)
#q2 = (27,40)
#print(z61.isElement(q1),z61.isElement(q2))
#print(z61.plus(q1,q2))

def str_to_bytes(s, byts = BIG_BYTES):
  ret = []
  for c in s:
    ret.append(ord(c))
  ret.append(0)
  while (len(ret) % byts != 0):
    ret.append(0)
  return ret

def bytes_to_nums(arr, byts = BIG_BYTES):
  #The numbers will be xored with a 192 bit number. Before being scrambled somewhat.
  arr2 = arr #Previously the padding was done here
  ret = []
  for i in range(len(arr2) // byts):
    #Combines bytes 0 through 23
    ii = i * byts
    next_num = arr2[ii]
    for j in range(ii+1, ii+byts):
      next_num = next_num << 8
      next_num += arr2[j]
    ret.append(next_num)
  return ret

def single_num_to_bytes(n, byts = BIG_BYTES):
  #More significant bytes are at the beginning of the list
  mask = 255
  ret = [0]*byts
  n2 = n
  for i in range(byts):
    next_num = n2 & mask
    ret[byts-1-i] = next_num
    #next_num = next_num >> 8 WRONG VARIABLE
    n2 = n2 >> 8
  return ret

def nums_to_bytes(arr, byts = BIG_BYTES):
  ret = []
  for n in arr:
    ret += single_num_to_bytes(n, byts)
  return ret

def bytes_to_str(arr):
  ret = ""
  for b in arr:
    if (b==0):
      return ret
    try:
      ret += chr(b)
    except:
      return ""
  return ret

def str_to_nums(s,byts=BIG_BYTES):
  return bytes_to_nums(str_to_bytes(s),byts)

def nums_to_str(arr,byts=BIG_BYTES):
  return bytes_to_str(nums_to_bytes(arr,byts))

copypasta = '''
Alas, poor Yorick! I knew him, Horatio: a fellow
of infinite jest, of most excellent fancy: he hath
borne me on his back a thousand times; and now, how
abhorred in my imagination it is! my gorge rims at
it. Here hung those lips that I have kissed I know
not how oft. Where be your gibes now? your
gambols? your songs? your flashes of merriment,
that were wont to set the table on a roar? Not one
now, to mock your own grinning? quite chap-fallen?
Now get you to my ladys chamber, and tell her, let
her paint an inch thick, to this favour she must
come; make her laugh at that.
'''

PRIME1 = 20448499098555461345475631188258768162283127532962587212863
PRIME2 = 24773275047705793803944818051912547746690447460616013205159
big_n = PRIME1 * PRIME2
PUBLIC_E = 65537
#prime1 and prime2 were used for previous testing

def rsa_power(arr, n, e):
  ret = []
  for i in arr:
    ret.append(power(i,e,n))
  return ret

def rsa_encrypt(s, n, e):
  return rsa_power(str_to_nums(s,48), n, e)

def rsa_decrypt(arr, n, d):
  return nums_to_str(rsa_power(arr,n,d),48)

def findPrivateKey(e, p1, p2):
  phi = (p1 - 1) * (p2 - 1)
  return inverse(e, phi)

def matrixMultiply(A, b, n = False): #I never used this
  a_width = len(b)
  a_height = len(A) // a_width
  ret = [0]*a_height
  for i in range(a_height):
    next_num=0
    for j in range(a_width):
      next_num += (A[(i*a_width) + j] * b[j])
    ret[i] = next_num
  if n:
    for i in a_height:
      ret[i] = ret[i] % n
  return ret

def identity(n):
  if (n==0):
    return []
  prev = identity(n-1)
  prev.append(n-1)
  return prev

def swap(arr,i,j):
  inter = arr[i]
  arr[i] = arr[j]
  arr[j] = inter

def scramble(n,key):
  #First decides what should be in index 0, then index 1, and so on
  arr = identity(n)
  for i in range(n):
    hashed = hash(f"{key}; {i}")
    j = i + (hashed%(n-i))
    swap(arr,i,j)
  return arr

def reverse_scramble(n,key):
  rev = scramble(n,key)
  arr = [0]*n
  for i in range(n):
    arr[rev[i]]=i
  return arr

def numarray_xor(arr, key, byts = BIG_BYTES):
  arr2 = []
  keey = key & ((1 << (8*byts)) - 1)
  #print(f"key is {key}, keey is {keey}")
  for num in arr:
    #print(f"num is {num} and the xored one is {num ^ keey}")
    arr2.append(num ^ keey)
  return arr2

def barray_xor(arr, key, byts = BIG_BYTES):
  return nums_to_bytes(numarray_xor(bytes_to_nums(arr,byts),key,byts),byts)

def modify_with_key(arr, key, reverse = False, byts = BIG_BYTES):
  #When encrypting, it should xor after scrambling. Takes in byte array, outputs byte array
  #The scrambling is so that if Eve knows that all messages start with the same 24 bytes, she can't just xor those with the ones she sees to find the key.
  #It also scrambles each 24-byte chunk a different way. That is because the key for scramble includes which chunk the algorithm is on.
  if reverse:
    #new_barr = barray_xor(arr,byts)
    #I SET BYTS AS THE KEY INSTEAD OF KEY
    new_barr = barray_xor(arr,key,byts)
  else:
    new_barr = arr
  newer_barr = []
  for i in range(len(arr) // byts):
    #sub_arr = new_barr[i*byts : (i+1):byts]
    #Seriously?
    sub_arr = new_barr[i*byts : (i+1)*byts]
    #print(f"sub_arr is now {sub_arr}")
    if reverse:
      scrambler = reverse_scramble(byts,f"{key}; {i}")
    else:
      scrambler = scramble(byts,f"{key}; {i}")
    #print(f"scrambler is now {scrambler}")
    sub_arr2 = []
    for j in range(byts):
      sub_arr2.append(sub_arr[scrambler[j]])
    newer_barr += sub_arr2
    #I did not mean to put the following in the for loop. That will soon be remedied.
  if reverse:
    return newer_barr
  else:
    #print(f"newer_barr, as a string, is {bytes_to_str(newer_barr)}\n\n")
    xored = barray_xor(newer_barr,key,byts)
    #print(f"After xor, it is {bytes_to_str(xored)}\n\n")
    return xored

def key_encrypt(s, key, byts = BIG_BYTES):
  return modify_with_key(str_to_bytes(s,byts), key, False, byts)

def key_decrypt(arr, key, byts = BIG_BYTES):
  return bytes_to_str(modify_with_key(arr, key, True, byts))

#ciphertext = key_encrypt(copypasta, PRIME1)
#print(ciphertext)
#deciphertext = key_decrypt(ciphertext, PRIME1)
#print(deciphertext)

class Person:
  #The methods with getPublic in their name are the ones that other people can access.
  #But Python doesn't have private methods, so just deal with it.
  def __init__(self,name):
    self.dict = {}
    self.group = None
    self.name = name
    self.sd("Nosy", (name == "Eve"))
    if USING_RSA:
      self.pickPrimes()
      self.pickExponent()
    else:
      self.pickMultiplier()

  
  def gd(self,s):
    if s in self.dict.keys():
      return self.dict[s]
    return None

  def sd(self,s,n):
    self.dict[s] = n

  def joinGroup(self,g):
    g.people.add(self)
    self.group = g
  
  def think(self,thought):
    print(f"{self.name} thinks: {thought}")

  def getPublicModulo(self):
    return self.gd("Prime 1") * self.gd("Prime 2")
  
  def getPublicExponent(self):
    return self.gd("Public Exponent")
  
  def isNosy(self):
    return self.gd("Nosy")
  
  def pickPrimes(self):
    self.sd("Prime 1", find_prime())
    self.sd("Prime 2", find_prime())
    self.think(f"My primes are {self.gd("Prime 1")} and {self.gd("Prime 2")}")
  
  def pickExponent(self, e = PUBLIC_E):
    self.sd("Public Exponent", e)
    self.think(f"My public exponent is {e}")
  
  def pickMultiplier(self):
    #I have no clue how big the private multipliers are supposed to be, I guess they just have to be big enough for discrete log problem to be hard
    d = largeInt(1 << 128) + (2 << 128)
    self.think(f"My private multiplier is {d}")
    self.sd("Private Multiplier", d)
  
  def getPublicQ(self):
    g = self.group.public_g
    d = self.gd("Private Multiplier")
    self.think(f"Someone wants my public Q. I should multiply {g} by my private multiplier {d}.")
    return self.group.ec.times(g,d)
  
  def exchangeKey(self,other):
    other_q = other.getPublicQ()
    self.think(f"So {other.name}'s public Q is {other_q}. I should multiply that by my private multiplier")
    k = self.group.ec.times(other_q,self.gd("Private Multiplier"))
    self.think(f"So the shared secret with {other.name} is {k}. I'll remember the first value for communication.")
    self.sd(f"Communication with {other.name}",k[0])
  
  def exchangeAllKeys(self):
    for other in self.group.people:
      if (other.name != self.name):
        self.exchangeKey(other)
  
  def receiveMessageRSA(self,ciphertext,intended_recipient):
    will_decrypt = False
    if (self.name == intended_recipient.name):
      self.think("This message is for me!")
      will_decrypt = True
    elif self.isNosy():
      self.think("I should eavesdrop!")
      will_decrypt = True
    if not will_decrypt:
      return
    d = findPrivateKey(self.getPublicExponent(), self.gd("Prime 1"), self.gd("Prime 2"))
    deciphertext = rsa_decrypt(ciphertext, self.getPublicModulo(), d)
    if (deciphertext == ""):
      self.think("That message was nonsense!")
    else:
      self.think(f"The message says, {deciphertext}")
  
  def sendMessageRSA(self,plaintext,intended_recipient):
    ir = intended_recipient
    ciphertext = rsa_encrypt(plaintext, ir.getPublicModulo(), ir.getPublicExponent())
    self.think(f"Sending a message. {ir.name}'s public modulus is {ir.getPublicModulo()} and their exponent is {ir.getPublicExponent()}")
    for other in self.group.people:
      if (other.name != self.name):
        other.receiveMessageRSA(ciphertext, ir)
  
  def receiveMessageECDH(self,ciphertext,sender,intended_recipient):
    will_decrypt = False
    if (self.name == intended_recipient.name):
      self.think(f"This message from {sender.name} is for me!")
      will_decrypt = True
    elif self.isNosy():
      self.think(f"I should eavesdrop on {sender.name}'s message!")
      will_decrypt = True
    if not will_decrypt:
      return
    key = self.gd(f"Communication with {sender.name}")
    self.think(f"Using key {key}...")
    deciphertext = key_decrypt(ciphertext, key)
    if (deciphertext == ""):
      self.think("That message was nonsense!")
    else:
      self.think(f"The message says, {deciphertext}")
  
  def sendMessageECDH(self,plaintext,intended_recipient):
    ir = intended_recipient
    key = self.gd(f"Communication with {ir.name}")
    self.think(f"To send a message to {ir.name}, I should use the key {key}")
    ciphertext = key_encrypt(plaintext,key)
    for other in self.group.people:
      if (other.name != self.name):
        other.receiveMessageECDH(ciphertext, self, ir)
  
  def __str__(self):
    return self.name


class Group:
  def __init__(self,arr):
    self.people = set()
    for person in arr:
      person.joinGroup(self)
    if not USING_RSA:
      self.decide_on_EC()
  
  def think(self,thought):
    print(f"The group is thinking: {thought}")
  
  def decide_on_EC(self):
    while True:
      modulus = find_prime()
      #192 bits or more.
      lin_coef = largeInt(modulus)
      con_coef = largeInt(modulus)
      new_ec = EllipticCurve(modulus,lin_coef,con_coef)
      if (new_ec.is_valid()):
        self.ec = new_ec
        self.public_g = new_ec.findRandomElement()
        self.think(f"Modulus is {modulus}, linear coefficient is {lin_coef}, constant coefficient is {con_coef}")
        self.think(f"Public element is {self.public_g}")
        return

if __name__ == "__main__":
  alice = Person("Alice")
  bob = Person("Bob")
  eve = Person("Eve")
  group1 = Group([alice,bob,eve])
  
  if USING_RSA:
    alice.sendMessageRSA(copypasta,bob)
  else:
    for person in [alice,bob,eve]:
      person.exchangeAllKeys()
    alice.sendMessageECDH(copypasta,bob)