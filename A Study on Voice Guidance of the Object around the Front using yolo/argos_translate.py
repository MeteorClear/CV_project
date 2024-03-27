import argostranslate
import argostranslate.package
import argostranslate.translate

'''
Source file for Argos translation
Used for offline translation
Argos Translate is Open-source offline translation library written in Python
https://github.com/argosopentech/argos-translate
'''

# Value to verify that the required package is installed
VALID_CHECK = False

def valid_check(package_num: str= None, \
                all: bool= False) -> None:
    '''
    Verify installation of packages for offline translation
    '''
    global VALID_CHECK

    # load and check language pack
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    # install all package (not recommended)
    if all:
        argostranslate.argospm.install_all_packages()

    elif package_num != None:
        argostranslate.package.install_from_path(available_packages[package_num].download())

    # Temporary installation of Korean-English(26 and 48)
    else:
        installed_languages = argostranslate.translate.get_installed_languages()
        install_lang = list(map(str, installed_languages))

        if not('English' in install_lang and 'Korean' in install_lang):
            argostranslate.package.install_from_path(available_packages[26].download())
            argostranslate.package.install_from_path(available_packages[48].download())

    VALID_CHECK = True
    return


def text_translate(text: str, \
                   dest: str, \
                   src: str, \
                   detail: bool= False) -> str:
    '''
    Translate given text, 
    Required text, target language and language of text
    '''
    global VALID_CHECK

    if not VALID_CHECK:
        valid_check()
    
    trans_text = argostranslate.translate.translate(text, src, dest)
    
    if detail:
        return (trans_text, text, dest, src, None, None)
    
    return trans_text