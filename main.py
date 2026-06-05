# main.py
# Single entry point for the SafeFrame system.
# Usage:
#   python main.py train       — train the model on your dataset
#   python main.py evaluate    — test accuracy + F1 + confusion matrix
#   python main.py moderate <image_path>  — moderate a single image
#   python main.py demo        — run demo on 5 images from uploads folder
#   python main.py server      — start the Flask web app on localhost:5000

import sys


def cmd_train():
    # WHAT: Run full training pipeline
    # WHY:  Lets user kick off training from one unified command
    # IN:   None
    # OUT:  None (saves model to safeframe_model.pth)
    from train import train
    train()


def cmd_evaluate():
    # WHAT: Evaluate saved model on test set
    # WHY:  Unified command — no need to remember which file does evaluation
    # IN:   None
    # OUT:  None (prints accuracy, F1, confusion matrix)
    from evaluate import evaluate
    evaluate()


def cmd_moderate(image_path: str):
    # WHAT: Moderate a single image from the command line
    # WHY:  Quick test — point at any image and see the result immediately
    # IN:   image_path (str) — path to image file
    # OUT:  None (prints result, saves output to outputs/results/)
    from moderate import moderate_image
    moderate_image(image_path)


def cmd_demo():
    # WHAT: Run the 5-image demo and print results table
    # WHY:  Monday demo entry point — one command shows the full pipeline
    # IN:   None
    # OUT:  None (prints table, saves outputs/demo_report.csv)
    from demo import run_demo
    run_demo()


def cmd_server():
    # WHAT: Start the Flask web app
    # WHY:  Lets user launch the full web UI from the same entry point
    # IN:   None
    # OUT:  None (blocks — runs Flask dev server on port 5000)
    print("[SERVER]   Starting SafeFrame web server on http://localhost:5000")
    print("[SERVER]   Press Ctrl+C to stop.\n")
    from app import app, get_engine
    get_engine()
    app.run(host="0.0.0.0", port=5000, debug=False)


def print_help():
    # WHAT: Print available commands when user runs main.py with no arguments
    # WHY:  Makes the project beginner-friendly — no need to read source code
    # IN:   None
    # OUT:  None
    print("\nSafeFrame — AI Content Moderation System")
    print("=" * 42)
    print("  python main.py train                  Train ResNet50 on your dataset")
    print("  python main.py evaluate               Test accuracy + F1 + confusion matrix")
    print("  python main.py moderate <image>       Moderate a single image")
    print("  python main.py demo                   Run 5-image demo pipeline")
    print("  python main.py server                 Start web app on localhost:5000")
    print()


COMMANDS = {
    "train":    cmd_train,
    "evaluate": cmd_evaluate,
    "demo":     cmd_demo,
    "server":   cmd_server,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS and sys.argv[1] != "moderate":
        print_help()
        sys.exit(0)

    command = sys.argv[1]

    if command == "moderate":
        if len(sys.argv) < 3:
            print("[ERROR]    Usage: python main.py moderate <image_path>")
            sys.exit(1)
        cmd_moderate(sys.argv[2])
    else:
        COMMANDS[command]()
