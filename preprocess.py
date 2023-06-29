from helper_functions import process

DATA_ROOT = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/vat_sat_subjects"
TARGET_ROOT = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/v1"
EXTENSION = ".jpg"
SIZE = (392, 363)
LIMIT = 50000

if __name__ == "__main__":
    process(DATA_ROOT, LIMIT ,TARGET_ROOT, SIZE, EXTENSION)