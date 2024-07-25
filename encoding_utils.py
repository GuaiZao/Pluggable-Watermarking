import random
import hamming_codec

def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
def bytearray_to_binstr(ba):
    return ''.join(bin(byte)[2:].zfill(8) for byte in ba)
def bin_str_to_bytearray(s):
    byte_array = bytearray([int(s[i:i + 8], 2) for i in range(0, len(s), 8)])
    return byte_array


def encode_BCH_fin(lenth,BCHcoder):
    item = (lenth//7)
    r = lenth%7
    fin_str = ''
    get_str = [0,3,5,6]
    for i in range(item):
        code = random.randint(0, 3)
        binary_str = bytearray([get_str[code]])
        ecc = BCHcoder.encode(binary_str)
        encoded_message = bytearray_to_binstr(binary_str)[5:8]+bytearray_to_binstr(ecc)[3:7]
        fin_str+=encoded_message
    for i in range(r):
        fin_str+='0'
    return fin_str


def decode_BCH_fin(lenth,fin,BCHcoder):
    item = (lenth//7)
    r = lenth%7
    fin_str = ''
    flag = list()
    getstr = bytearray([0,3,5,6])
    # flag = 0
    for i in range(item):
        encoded_message = fin[i * 7:(i + 1) * 7]
        code = '00000'+encoded_message[0:3]
        ecc = '000'+encoded_message[3:7]+'0'
        # en_data = bin_str_to_bytearray(code)+bin_str_to_bytearray(ecc)
        # source = source_fin[i*7:(i+1)*7]
        num, data_corrected, ecc_en = BCHcoder.decode(bin_str_to_bytearray(code), bin_str_to_bytearray(ecc))
        if num==-1 or data_corrected not in getstr:
            fin_str+=encoded_message
            num = -1
        else:
            data_corrected_str = bytearray_to_binstr(data_corrected)
            ecc_en_str = bytearray_to_binstr(ecc_en)
            fin_str+=data_corrected_str[5:8]+ecc_en_str[3:7]
        flag.append(num)

    return fin_str, flag


def encode_hamming_fin(lenth):
    item = (lenth//7)
    r = lenth%7
    fin_str = ''
    for i in range(item):
        binary_str = ''.join(random.choice('01') for _ in range(4))
        encoded_message = hamming_codec.encode(int(binary_str, 2), 4)
        fin_str+=encoded_message
    for i in range(r):
        fin_str+='0'
    return fin_str

def decode_hamming_fin(lenth,fin,source_fin):
    item = (lenth//7)
    r = lenth%7
    fin_str = ''
    flag = 0
    for i in range(item):
        encoded_message = fin[i*7:(i+1)*7]
        res = hamming_codec.decode(int(encoded_message, 2), len(encoded_message))
        source = source_fin[i*7:(i+1)*7]
        res_message = hamming_codec.encode(int(res, 2), 4)
        hamming_distance = sum(el1 != el2 for el1, el2 in zip(res_message, source))
        if hamming_distance>1:
            fin_str+=encoded_message
            flag+=1
        else :
            fin_str+=res_message
    return fin_str,flag
        # print(binary_str)