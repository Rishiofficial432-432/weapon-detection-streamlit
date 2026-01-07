from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # We use 'mps' for Apple Silicon acceleration if available, otherwise 'cpu'
    # device='mps' is often auto-detected by YOLO, but we can specify if needed.
    # 10 epochs for demonstration/quick result.
    results = model.train(
        data='custom_data.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        name='custom_weapon_model_v2'
    )
    
    # Export the model
    success = model.export(format='pt')
    print(f"Training complete. Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train()
