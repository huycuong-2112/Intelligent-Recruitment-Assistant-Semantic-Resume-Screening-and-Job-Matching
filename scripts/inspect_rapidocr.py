import inspect
import rapidocr
print('rapidocr module:', rapidocr)
print('attributes:')
for name in sorted(dir(rapidocr)):
    print(name)

# try to find callable OCR entrypoints
for name in ['ocr', 'main', 'read', 'py_ocr', 'rapid_ocr', 'RapidOCR', 'readtext']:
    if hasattr(rapidocr, name):
        obj = getattr(rapidocr, name)
        print('\nFound candidate:', name, '->', obj)
        try:
            print('callable:', callable(obj))
        except Exception as e:
            print('call check failed', e)
