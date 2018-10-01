# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def nested_get(dic, keys):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    return dic[keys[-1]]

def dictionary_to_dir_string(dictionary):
    string = ""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            string += dictionary_to_dir_string(value)
        else:
            string += str(key) + "_" + str(value)

    string.replace(".","_")
    string.replace("'","_")
    string.replace(":","_")
    string.replace("}","_")
    string.replace("{","_")
    return string
