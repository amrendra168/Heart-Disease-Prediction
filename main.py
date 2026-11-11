from train_model import train_model
from predict import predict_heart_disease

def main():
    while True:
        print("\n========== HEART DISEASE PREDICTION ==========")
        print("1. Train Model")
        print("2. Make Prediction")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            train_model()

        elif choice == "2":
            predict_heart_disease()

        elif choice == "0":
            print("Thank you for using the project!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
