from src.models.classifier import initialize_classifier

def main():
    model, device = initialize_classifier()
    print(model)
    print("Model initialized on device:", device)

if __name__ == "__main__":
    main()
