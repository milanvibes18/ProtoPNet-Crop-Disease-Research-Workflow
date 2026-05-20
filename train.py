from model import Model

# =============================================================================
# STAGE 3: ProtoPNet Training Execution on PlantWild
# =============================================================================

if __name__ == "__main__":
    print("Initializing ProtoPNet Pipeline...")
    
    # 1. Initialize the model 
    # This automatically applies your config, seeds the environment, and builds the DataLoaders
    m = Model()
    
    # 2. Print out the architecture summary to confirm 98 classes and parameters
    m.summary()
    
    # 3. Start the 4-Phase Training Loop
    m.train()
    
    # 4. Save the final weights once training is completely finished
    m.save()
    
    # 5. Evaluate on the untouched, real test set
    print("\nRunning final evaluation on the test set...")
    m.evaluate("test")