from dicom import DICOM


def main():
    file = "DICOM/digest_article"
    dicom = DICOM(file)
    data = dicom.process_dynamic("data", plot=True)


if __name__ == "__main__":
    main()
