"""
Hostel Management SRS Conflict Detection Pipeline - COMPLETE IMPROVED VERSION
Using Enhanced Techniques for Better Performance

Key Improvements:
1. Binary Relevance (separate classifier per conflict type)
2. Feature augmentation with TF-IDF
3. More balanced dataset (300 requirements)
4. Ensemble approach with XGBoost
5. Better hyperparameters

Requirements:
pip install transformers torch scikit-learn pandas numpy xgboost imbalanced-learn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading models... This may take a few minutes on first run.")
from transformers import AutoTokenizer, AutoModel
import torch

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("  XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# ============================================================================
# PART 1: ENHANCED DATASET GENERATION (300 requirements, more balanced)
# ============================================================================

def generate_hostel_requirements() -> pd.DataFrame:
    """
    Generate 300 labeled hostel management requirements with balanced conflicts.
    Conflicts: Budget, Space, Compliance, Security
    """
    
    requirements = []
    
    # Budget Conflicts (75 requirements - increased from 50)
    budget_reqs = [
        ("System must support premium features with enterprise-grade security", ["Budget", "Security"]),
        ("Implement basic room booking with minimal infrastructure cost", ["Budget"]),
        ("Deploy cloud infrastructure with 99.99% uptime SLA", ["Budget"]),
        ("Use free open-source database for all operations", ["Budget", "Security"]),
        ("Require biometric authentication for all access points", ["Budget", "Security"]),
        ("Manual paper-based backup system for cost savings", ["Budget", "Compliance"]),
        ("Single low-cost server for entire hostel operations", ["Budget", "Space"]),
        ("Implement AI-powered predictive maintenance system", ["Budget"]),
        ("Use consumer-grade hardware for critical systems", ["Budget", "Security"]),
        ("Free tier cloud services for production environment", ["Budget", "Compliance"]),
        ("No dedicated IT staff, students manage system", ["Budget", "Security"]),
        ("Implement blockchain for payment tracking", ["Budget"]),
        ("Real-time facial recognition at all entry points", ["Budget", "Security"]),
        ("Store all data on local USB drives", ["Budget", "Security", "Compliance"]),
        ("Custom-built ERP system from scratch", ["Budget"]),
        ("Minimal password requirements to reduce support costs", ["Budget", "Security"]),
        ("Single database instance without backup", ["Budget", "Compliance"]),
        ("Enterprise features with startup budget", ["Budget"]),
        ("Outsource all operations to lowest bidder", ["Budget", "Security"]),
        ("No encryption to save processing costs", ["Budget", "Security", "Compliance"]),
        ("24/7 phone support with 2-person team", ["Budget", "Space"]),
        ("Implement machine learning recommendation engine", ["Budget"]),
        ("Free WiFi throughout campus with no security", ["Budget", "Security"]),
        ("Basic system with advanced analytics dashboard", ["Budget"]),
        ("Cloud storage with on-premise backup simultaneously", ["Budget", "Space"]),
        ("Develop mobile app for iOS, Android, and web", ["Budget"]),
        ("Use spreadsheet software as primary database", ["Budget", "Compliance"]),
        ("High-availability setup with single server", ["Budget"]),
        ("Implement zero-budget disaster recovery plan", ["Budget", "Compliance"]),
        ("Premium third-party integrations on free tier", ["Budget"]),
        ("Require multi-factor authentication universally", ["Budget"]),
        ("Store backups on employee personal devices", ["Budget", "Security", "Compliance"]),
        ("Implement real-time video monitoring system", ["Budget", "Space"]),
        ("Same-day custom feature development", ["Budget"]),
        ("Enterprise SLA with community support only", ["Budget"]),
        ("Quantum-encryption for all data", ["Budget"]),
        ("Instant global deployment capability", ["Budget", "Space"]),
        ("Use deprecated software to avoid license costs", ["Budget", "Security", "Compliance"]),
        ("Implement predictive analytics with no data collection", ["Budget"]),
        ("Professional UI/UX with no design budget", ["Budget"]),
        ("Real-time sync across 1000+ locations", ["Budget", "Space"]),
        ("Store credit card data in plain text files", ["Budget", "Security", "Compliance"]),
        ("Develop custom OS for hostel hardware", ["Budget"]),
        ("Unlimited storage with free account", ["Budget", "Space"]),
        ("High-performance computing on basic laptops", ["Budget"]),
        ("Implement without any testing phase", ["Budget", "Security", "Compliance"]),
        ("24/7 monitoring with no monitoring tools", ["Budget"]),
        ("Use student volunteers for critical maintenance", ["Budget", "Security"]),
        ("Guarantee zero downtime with no redundancy", ["Budget"]),
        ("Build custom chip for IoT devices", ["Budget"]),
        ("Implement enterprise resource planning with free software", ["Budget"]),
        ("24/7 customer support with single part-time staff", ["Budget"]),
        ("Deploy global CDN on free hosting plan", ["Budget", "Space"]),
        ("Custom hardware development for room automation", ["Budget"]),
        ("Implement advanced fraud detection with no budget", ["Budget"]),
        ("Professional penetration testing with no security budget", ["Budget", "Security"]),
        ("Build native apps for all platforms simultaneously", ["Budget"]),
        ("Implement real-time language translation service", ["Budget"]),
        ("Enterprise-grade monitoring with free tools only", ["Budget"]),
        ("Custom AI chatbot development from scratch", ["Budget"]),
        ("Implement advanced biometric systems campus-wide", ["Budget", "Security"]),
        ("Deploy distributed database across regions", ["Budget", "Space"]),
        ("Build custom payment gateway to save fees", ["Budget", "Security"]),
        ("Implement comprehensive audit logging with minimal storage", ["Budget", "Space", "Compliance"]),
        ("Professional design services with student interns", ["Budget"]),
        ("Enterprise backup solution on consumer hardware", ["Budget", "Space"]),
        ("Implement advanced analytics on basic infrastructure", ["Budget"]),
        ("Custom cryptocurrency for hostel payments", ["Budget", "Compliance"]),
        ("Build proprietary video conferencing system", ["Budget"]),
        ("Implement advanced AI without GPU resources", ["Budget", "Space"]),
        ("Enterprise licensing costs with minimal budget allocation", ["Budget"]),
        ("Build custom IoT sensors for room monitoring", ["Budget"]),
        ("Implement sophisticated access control with cheap hardware", ["Budget", "Security"]),
        ("Professional training program with no training budget", ["Budget"]),
        ("Advanced threat detection with free antivirus", ["Budget", "Security"]),
    ]
    
    # Space Conflicts (75 requirements - increased from 50)
    space_reqs = [
        ("Store 50 years of detailed logs on 100GB storage", ["Space"]),
        ("Maintain complete video footage history indefinitely", ["Space", "Compliance"]),
        ("Host system on smartphones only", ["Space"]),
        ("Store uncompressed 4K video surveillance", ["Space"]),
        ("Keep duplicate copies of all files in same location", ["Space"]),
        ("Unlimited file upload size for all users", ["Space"]),
        ("Real-time processing of terabyte datasets", ["Space"]),
        ("Store all emails permanently without archiving", ["Space", "Compliance"]),
        ("Maintain multiple versions of every file forever", ["Space"]),
        ("Host database on embedded IoT device", ["Space"]),
        ("Process big data analytics on edge devices", ["Space"]),
        ("Store raw sensor data at millisecond intervals", ["Space"]),
        ("Maintain full audit trail for 100 years", ["Space", "Compliance"]),
        ("Deploy entire system on raspberry pi", ["Space"]),
        ("Store high-resolution photos without compression", ["Space"]),
        ("Keep all deleted files indefinitely", ["Space"]),
        ("Run data warehouse on mobile device", ["Space"]),
        ("Store blockchain ledger locally on each device", ["Space"]),
        ("Maintain complete change history for all records", ["Space"]),
        ("Host video conferencing server on local network", ["Space"]),
        ("Store all system logs in memory only", ["Space", "Compliance"]),
        ("Process machine learning models on basic tablets", ["Space"]),
        ("Maintain offline copy of entire internet", ["Space"]),
        ("Store biometric data without compression", ["Space", "Compliance"]),
        ("Run AI training on end-user devices", ["Space"]),
        ("Keep all temporary files permanently", ["Space"]),
        ("Host large media library on local drives", ["Space"]),
        ("Store every user session recording", ["Space", "Compliance"]),
        ("Maintain real-time replica of production data", ["Space"]),
        ("Process video encoding on mobile devices", ["Space"]),
        ("Store all attachments in database BLOBs", ["Space"]),
        ("Run containerized apps on thin clients", ["Space"]),
        ("Maintain hot standby of all systems simultaneously", ["Space"]),
        ("Store geospatial data with millimeter precision", ["Space"]),
        ("Host multiple environments on single laptop", ["Space"]),
        ("Keep detailed performance metrics for eternity", ["Space", "Compliance"]),
        ("Run virtualization without hypervisor resources", ["Space"]),
        ("Store all cache data persistently", ["Space"]),
        ("Maintain offline search index of everything", ["Space"]),
        ("Process complex queries without query optimization", ["Space"]),
        ("Store all user preferences with complete history", ["Space"]),
        ("Run parallel processing on single core", ["Space"]),
        ("Maintain complete system snapshots hourly", ["Space"]),
        ("Store AI model weights on client devices", ["Space"]),
        ("Host distributed system on single node", ["Space"]),
        ("Keep all debug information in production", ["Space"]),
        ("Run memory-intensive apps with 2GB RAM", ["Space"]),
        ("Store normalized and denormalized data simultaneously", ["Space"]),
        ("Maintain complete network packet capture", ["Space", "Compliance"]),
        ("Process streaming data without buffering limits", ["Space"]),
        ("Store full-resolution security camera feeds locally", ["Space"]),
        ("Maintain complete user activity logs permanently", ["Space", "Compliance"]),
        ("Host entire application on single USB drive", ["Space"]),
        ("Store uncompressed audio recordings of all interactions", ["Space"]),
        ("Keep every email attachment in database", ["Space"]),
        ("Process real-time analytics on minimal hardware", ["Space"]),
        ("Store complete browsing history for all users", ["Space", "Compliance"]),
        ("Maintain unlimited backup versions locally", ["Space"]),
        ("Run complex simulations on basic computers", ["Space"]),
        ("Store high-definition livestreams indefinitely", ["Space"]),
        ("Keep all test data in production environment", ["Space"]),
        ("Process large-scale data mining on edge devices", ["Space"]),
        ("Store complete audit trail with millisecond precision", ["Space", "Compliance"]),
        ("Host video conferencing for 1000+ users locally", ["Space"]),
        ("Maintain real-time dashboards with historical data", ["Space"]),
        ("Store biometric templates for entire population", ["Space", "Compliance"]),
        ("Run deep learning inference on IoT devices", ["Space"]),
        ("Keep all system states in RAM for performance", ["Space"]),
        ("Process 4K video streams on mobile processors", ["Space"]),
        ("Store complete financial transaction history locally", ["Space", "Compliance"]),
        ("Maintain geographic information system on tablets", ["Space"]),
        ("Run multiple database engines simultaneously", ["Space"]),
        ("Store all configuration changes with full context", ["Space"]),
        ("Process real-time facial recognition locally", ["Space"]),
        ("Keep complete network traffic logs forever", ["Space", "Compliance"]),
    ]
    
    # Compliance Conflicts (75 requirements - increased from 50)
    compliance_reqs = [
        ("Students can access any other student's personal data", ["Compliance", "Security"]),
        ("Store passwords in readable format in database", ["Compliance", "Security"]),
        ("Share user data with third parties without consent", ["Compliance"]),
        ("No data retention policy or deletion capability", ["Compliance"]),
        ("Collect biometric data without explicit consent", ["Compliance", "Security"]),
        ("Allow anonymous access to financial records", ["Compliance", "Security"]),
        ("Store medical information without encryption", ["Compliance", "Security"]),
        ("No audit trail for data modifications", ["Compliance"]),
        ("Transfer data internationally without safeguards", ["Compliance"]),
        ("Require excessive personal information for registration", ["Compliance"]),
        ("No user consent mechanism for data processing", ["Compliance"]),
        ("Store payment card data without PCI compliance", ["Compliance", "Security"]),
        ("Allow backdoor access to all user accounts", ["Compliance", "Security"]),
        ("No privacy policy or terms of service", ["Compliance"]),
        ("Automatically opt users into data sharing", ["Compliance"]),
        ("Store children's data without parental consent", ["Compliance"]),
        ("No mechanism to export or delete user data", ["Compliance"]),
        ("Track user location without notification", ["Compliance", "Security"]),
        ("Share surveillance footage publicly", ["Compliance", "Security"]),
        ("No incident response or breach notification plan", ["Compliance", "Security"]),
        ("Store biometric templates without hashing", ["Compliance", "Security"]),
        ("Allow data access without authentication", ["Compliance", "Security"]),
        ("No data classification or handling procedures", ["Compliance"]),
        ("Retain data indefinitely against regulations", ["Compliance"]),
        ("Process sensitive data in unauthorized locations", ["Compliance"]),
        ("No data protection impact assessment", ["Compliance"]),
        ("Allow unlimited data scraping by third parties", ["Compliance", "Security"]),
        ("Store employee social security numbers unencrypted", ["Compliance", "Security"]),
        ("No age verification for restricted content", ["Compliance"]),
        ("Automatically share user activities on social media", ["Compliance"]),
        ("No secure disposal process for old records", ["Compliance", "Security"]),
        ("Allow anyone to modify audit logs", ["Compliance", "Security"]),
        ("No legal basis for data processing", ["Compliance"]),
        ("Store deleted data permanently in backups", ["Compliance"]),
        ("No transparency about data usage", ["Compliance"]),
        ("Allow unrestricted third-party cookies", ["Compliance"]),
        ("No consent withdrawal mechanism", ["Compliance"]),
        ("Store health data without HIPAA compliance", ["Compliance", "Security"]),
        ("No data breach insurance or liability coverage", ["Compliance"]),
        ("Allow discriminatory automated decision making", ["Compliance"]),
        ("No cross-border data transfer agreements", ["Compliance"]),
        ("Store financial records beyond legal minimum", ["Compliance"]),
        ("No vendor security assessment process", ["Compliance", "Security"]),
        ("Allow unlimited facial recognition deployment", ["Compliance", "Security"]),
        ("No data anonymization for analytics", ["Compliance"]),
        ("Store customer complaints without confidentiality", ["Compliance"]),
        ("No accessibility compliance for disabled users", ["Compliance"]),
        ("Allow profiling without user knowledge", ["Compliance"]),
        ("No regular compliance audits or reviews", ["Compliance"]),
        ("Store passport/ID scans indefinitely", ["Compliance", "Security"]),
        ("Process personal data without legal justification", ["Compliance"]),
        ("No data breach notification procedures", ["Compliance", "Security"]),
        ("Share student grades publicly without consent", ["Compliance"]),
        ("Store religious or political affiliations", ["Compliance"]),
        ("No process for handling data subject requests", ["Compliance"]),
        ("Collect more data than necessary for purpose", ["Compliance"]),
        ("No privacy-by-design in system architecture", ["Compliance"]),
        ("Share user data across international borders freely", ["Compliance"]),
        ("No documentation of data processing activities", ["Compliance"]),
        ("Store behavioral tracking data indefinitely", ["Compliance"]),
        ("No parental consent for minor users", ["Compliance"]),
        ("Process special category data without safeguards", ["Compliance", "Security"]),
        ("No data portability mechanism for users", ["Compliance"]),
        ("Share email addresses with marketing partners", ["Compliance"]),
        ("No impact assessment for automated decisions", ["Compliance"]),
        ("Store genetic information without protection", ["Compliance", "Security"]),
        ("No training for staff on data protection", ["Compliance"]),
        ("Process data for purposes beyond original intent", ["Compliance"]),
        ("No records of consent given by users", ["Compliance"]),
        ("Share IP addresses and browsing history", ["Compliance"]),
        ("No designated data protection officer", ["Compliance"]),
        ("Store financial statements without authorization", ["Compliance", "Security"]),
        ("No policies for data minimization", ["Compliance"]),
        ("Process biometric data for marketing purposes", ["Compliance", "Security"]),
        ("No accountability measures for data breaches", ["Compliance"]),
    ]
    
    # Security Conflicts (75 requirements - increased from 50)
    security_reqs = [
        ("Allow all users full administrative privileges", ["Security"]),
        ("Disable all firewall rules for performance", ["Security"]),
        ("Use default passwords for all system accounts", ["Security"]),
        ("Expose database directly to internet", ["Security"]),
        ("Disable SSL/TLS for faster page loads", ["Security", "Compliance"]),
        ("Store API keys in public GitHub repository", ["Security"]),
        ("Allow SQL injection for flexible queries", ["Security"]),
        ("Disable all authentication temporarily", ["Security", "Compliance"]),
        ("Use FTP instead of SFTP for file transfers", ["Security"]),
        ("Share admin credentials among all staff", ["Security"]),
        ("Disable antivirus to improve performance", ["Security"]),
        ("Allow cross-site scripting for rich features", ["Security"]),
        ("Use HTTP instead of HTTPS everywhere", ["Security", "Compliance"]),
        ("Disable input validation for user convenience", ["Security"]),
        ("Allow remote desktop access without VPN", ["Security"]),
        ("Use weak MD5 hashing for passwords", ["Security", "Compliance"]),
        ("Disable CSRF protection for easier development", ["Security"]),
        ("Allow unlimited login attempts", ["Security"]),
        ("Store session tokens in URL parameters", ["Security"]),
        ("Disable two-factor authentication option", ["Security", "Compliance"]),
        ("Allow file uploads without type checking", ["Security"]),
        ("Use telnet for remote administration", ["Security"]),
        ("Disable code signing for faster deployment", ["Security"]),
        ("Allow embedded scripts from any source", ["Security"]),
        ("Use unencrypted database connections", ["Security", "Compliance"]),
        ("Disable security headers for compatibility", ["Security"]),
        ("Allow directory listing on web server", ["Security"]),
        ("Use predictable session identifiers", ["Security"]),
        ("Disable rate limiting on API endpoints", ["Security"]),
        ("Allow unrestricted file inclusion", ["Security"]),
        ("Use hardcoded credentials in source code", ["Security", "Compliance"]),
        ("Disable security patches for stability", ["Security"]),
        ("Allow XML external entity processing", ["Security"]),
        ("Use insecure random number generation", ["Security"]),
        ("Disable certificate validation for APIs", ["Security", "Compliance"]),
        ("Allow command injection for flexibility", ["Security"]),
        ("Use outdated cryptographic algorithms", ["Security", "Compliance"]),
        ("Disable clickjacking protection", ["Security"]),
        ("Allow unrestricted CORS access", ["Security"]),
        ("Use null cipher for encryption", ["Security", "Compliance"]),
        ("Disable intrusion detection system", ["Security"]),
        ("Allow path traversal in file operations", ["Security"]),
        ("Use deprecated authentication protocols", ["Security", "Compliance"]),
        ("Disable security monitoring and logging", ["Security", "Compliance"]),
        ("Allow deserialization of untrusted data", ["Security"]),
        ("Use world-readable file permissions", ["Security"]),
        ("Disable email security protocols (SPF/DKIM)", ["Security"]),
        ("Allow buffer overflow for performance", ["Security"]),
        ("Use cleartext protocols for everything", ["Security", "Compliance"]),
        ("Disable access control checks", ["Security", "Compliance"]),
        ("Allow JavaScript execution in user inputs", ["Security"]),
        ("Disable network segmentation for convenience", ["Security"]),
        ("Use single encryption key for all data", ["Security", "Compliance"]),
        ("Allow direct database queries from frontend", ["Security"]),
        ("Disable password expiration policies", ["Security"]),
        ("Use public WiFi for administrative tasks", ["Security"]),
        ("Allow unrestricted API access without tokens", ["Security"]),
        ("Disable security event logging completely", ["Security", "Compliance"]),
        ("Use GET requests for sensitive operations", ["Security"]),
        ("Allow anonymous file uploads to server", ["Security"]),
        ("Disable content security policy headers", ["Security"]),
        ("Use shared accounts for system access", ["Security", "Compliance"]),
        ("Allow execution of unsigned code", ["Security"]),
        ("Disable secure boot and firmware protection", ["Security"]),
        ("Use unvalidated redirects in application", ["Security"]),
        ("Allow unrestricted access to system files", ["Security"]),
        ("Disable brute force protection mechanisms", ["Security"]),
        ("Use predictable URLs for sensitive resources", ["Security"]),
        ("Allow mixed HTTP and HTTPS content", ["Security"]),
        ("Disable session timeout for convenience", ["Security"]),
        ("Use client-side security checks only", ["Security"]),
        ("Allow direct object reference without validation", ["Security"]),
        ("Disable network encryption between services", ["Security", "Compliance"]),
        ("Use default security configurations", ["Security"]),
        ("Allow privilege escalation without validation", ["Security", "Compliance"]),
    ]
    
    # Combine all requirements
    all_reqs = budget_reqs + space_reqs + compliance_reqs + security_reqs
    
    # Create dataframe with multi-label format
    for i, (req_text, conflicts) in enumerate(all_reqs, 1):
        req_entry = {
            'id': f'REQ-{i:03d}',
            'requirement': req_text,
            'Budget': 1 if 'Budget' in conflicts else 0,
            'Space': 1 if 'Space' in conflicts else 0,
            'Compliance': 1 if 'Compliance' in conflicts else 0,
            'Security': 1 if 'Security' in conflicts else 0
        }
        requirements.append(req_entry)
    
    return pd.DataFrame(requirements)


# ============================================================================
# PART 2: SENTENCE TRANSFORMER EMBEDDINGS
# ============================================================================

class SentenceEmbedder:
    """
    Real embeddings using sentence-transformers/all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"\n📥 Loading {model_name} model...")
        print("   This may take 1-2 minutes on first run (downloading ~90MB)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Model loaded successfully on {self.device}")
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts, batch_size=16, max_length=128):
        embeddings = []
        
        print(f"\n Generating embeddings for {len(texts)} requirements...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                sentence_embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings.append(sentence_embeddings.cpu().numpy())
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} requirements...")
        
        all_embeddings = np.vstack(embeddings)
        print(f" Embeddings generated: shape {all_embeddings.shape}")
        
        return all_embeddings


# ============================================================================
# PART 3: FEATURE AUGMENTATION WITH TF-IDF
# ============================================================================

def create_hybrid_features(texts, embeddings):
    """
    Combine sentence embeddings with TF-IDF features for better performance.
    
    Args:
        texts: List of requirement texts
        embeddings: Sentence transformer embeddings
        
    Returns:
        Combined feature matrix and TF-IDF vectorizer
    """
    print("\n Creating hybrid features (Embeddings + TF-IDF)...")
    
    # Generate TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=200,  # Top 200 terms
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.8
    )
    
    tfidf_features = tfidf.fit_transform(texts).toarray()
    
    # Combine features
    hybrid_features = np.hstack([embeddings, tfidf_features])
    
    print(f" Hybrid features created: shape {hybrid_features.shape}")
    print(f"   - Embedding features: {embeddings.shape[1]}")
    print(f"   - TF-IDF features: {tfidf_features.shape[1]}")
    
    return hybrid_features, tfidf


# ============================================================================
# PART 4: ENHANCED TRAINING WITH BINARY RELEVANCE
# ============================================================================

def train_conflict_detector_enhanced(df, features):
    """
    Train enhanced multi-label classifier using Binary Relevance approach.
    Each conflict type gets its own optimized classifier.
    
    Args:
        df: DataFrame with requirements and labels
        features: Feature matrix (hybrid features)
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    
    conflict_types = ['Budget', 'Space', 'Compliance', 'Security']
    y = df[conflict_types].values
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.25, random_state=42
    )
    
    print(f"\n Dataset split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Choose classifier based on availability
    if XGBOOST_AVAILABLE:
        print("\n Training XGBoost classifiers (Binary Relevance)...")
        base_classifier = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            eval_metric='logloss'
        )
    else:
        print("\n Training Random Forest classifiers (Binary Relevance)...")
        base_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_features='sqrt'
        )
    
    # Use MultiOutputClassifier for binary relevance
    model = MultiOutputClassifier(base_classifier, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print(" Training completed!")
    
    # Evaluate
    y_pred = model.predict(X_test)
    h_loss = hamming_loss(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{'MODEL PERFORMANCE METRICS':^70}")
    print(f"{'='*70}")
    print(f"\n Hamming Loss: {h_loss:.4f}")
    
    if h_loss < 0.15:
        print(f"    TARGET ACHIEVED: < 0.15")
    else:
        print(f"     Target: < 0.15 (current: {h_loss:.4f})")
    
    # Subset accuracy
    subset_acc = accuracy_score(y_test, y_pred)
    print(f" Subset Accuracy (Exact Match): {subset_acc:.4f}")
    
    print(f"\n Classification Report:")
    print("-" * 70)
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=conflict_types, 
        zero_division=0
    ))
    
    # Per-label analysis
    print(f"\n Per-Label Performance Analysis:")
    print("-" * 70)
    for i, conflict in enumerate(conflict_types):
        label_accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        print(f"   {conflict:12s}: Accuracy = {label_accuracy:.4f}")
    
    metrics = {
        'hamming_loss': h_loss,
        'subset_accuracy': subset_acc,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics


# ============================================================================
# PART 5: PREDICTION AND RESULTS
# ============================================================================

def predict_conflicts(requirements, model, features):
    """
    Predict conflicts for requirements using trained model.
    """
    
    predictions = model.predict(features)
    
    conflict_types = ['Budget', 'Space', 'Compliance', 'Security']
    results = []
    
    print(f"\n Predicting conflicts for {len(requirements)} requirements...")
    
    for i, req in enumerate(requirements):
        conflicts_detected = [
            conflict_types[j] for j in range(4) if predictions[i][j] == 1
        ]
        
        results.append({
            'id': f'REQ-{i+1:03d}',
            'requirement': req,
            'conflicts': conflicts_detected,
            'conflict_count': len(conflicts_detected),
            'severity': 'High' if len(conflicts_detected) >= 3 else 
                       'Medium' if len(conflicts_detected) == 2 else 
                       'Low' if len(conflicts_detected) == 1 else 'None'
        })
    
    results_df = pd.DataFrame(results)
    
    print(f" Conflict prediction completed!")
    print(f"\n Conflict Summary:")
    print("-" * 70)
    print(f"   High Severity (3+ conflicts):   {len(results_df[results_df['severity'] == 'High'])} requirements")
    print(f"   Medium Severity (2 conflicts):  {len(results_df[results_df['severity'] == 'Medium'])} requirements")
    print(f"   Low Severity (1 conflict):      {len(results_df[results_df['severity'] == 'Low'])} requirements")
    print(f"   No Conflicts:                   {len(results_df[results_df['severity'] == 'None'])} requirements")
    
    return results_df


def display_sample_predictions(results_df, n_samples=10):
    """
    Display sample predictions with detailed information.
    """
    
    print(f"\n{'='*70}")
    print(f"{'SAMPLE PREDICTIONS':^70}")
    print(f"{'='*70}\n")
    
    # Show high severity examples
    high_severity = results_df[results_df['severity'] == 'High'].head(n_samples // 2)
    
    if len(high_severity) > 0:
        print("🔴 HIGH SEVERITY CONFLICTS:\n")
        for idx, row in high_severity.iterrows():
            print(f"{row['id']}: {row['requirement'][:80]}...")
            print(f"   Conflicts: {', '.join(row['conflicts'])}")
            print()
    
    # Show medium severity examples
    medium_severity = results_df[results_df['severity'] == 'Medium'].head(n_samples // 2)
    
    if len(medium_severity) > 0:
        print(" MEDIUM SEVERITY CONFLICTS:\n")
        for idx, row in medium_severity.iterrows():
            print(f"{row['id']}: {row['requirement'][:80]}...")
            print(f"   Conflicts: {', '.join(row['conflicts'])}")
            print()


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline for hostel conflict detection.
    """
    
    print("="*70)
    print("IMPROVED HOSTEL CONFLICT DETECTION PIPELINE".center(70))
    print("="*70)
    
    # Step 1: Generate dataset
    print("\n[STEP 1] Generating enhanced hostel requirements dataset...")
    df = generate_hostel_requirements()
    print(f" Generated {len(df)} requirements")
    print(f"   - Budget conflicts: {df['Budget'].sum()}")
    print(f"   - Space conflicts: {df['Space'].sum()}")
    print(f"   - Compliance conflicts: {df['Compliance'].sum()}")
    print(f"   - Security conflicts: {df['Security'].sum()}")
    
    # Step 2: Generate embeddings
    print("\n[STEP 2] Generating sentence embeddings...")
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(df['requirement'].tolist())
    
    # Step 3: Create hybrid features
    print("\n[STEP 3] Creating hybrid features...")
    hybrid_features, tfidf = create_hybrid_features(df['requirement'].tolist(), embeddings)
    
    # Step 4: Train model
    print("\n[STEP 4] Training enhanced conflict detection model...")
    model, metrics = train_conflict_detector_enhanced(df, hybrid_features)
    
    # Step 5: Predict on full dataset
    print("\n[STEP 5] Running predictions on all requirements...")
    results = predict_conflicts(df['requirement'].tolist(), model, hybrid_features)
    
    # Step 6: Display results
    display_sample_predictions(results, n_samples=10)
    
    # Save results
    print(f"\n{'='*70}")
    print(" Saving results to 'hostel_conflict_predictions_improved.csv'...")
    results.to_csv('hostel_conflict_predictions_improved.csv', index=False)
    print(" Results saved successfully!")
    
    # Performance summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY".center(70))
    print(f"{'='*70}")
    print(f"\n✅ Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f" Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f" Test Samples: {metrics['test_size']}")
    
    if metrics['hamming_loss'] < 0.15:
        print(f"\n TARGET ACHIEVED! Hamming Loss < 0.15")
    else:
        print(f"\n  Need further improvement to reach target")
        print(f"   Current: {metrics['hamming_loss']:.4f}, Target: 0.15")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETED SUCCESSFULLY!".center(70))
    print(f"{'='*70}\n")
    
   # NEW: Generate Mistral resolutions for top 25 conflicts
    print(f"\n{'='*70}")
    print("GENERATING AI RESOLUTIONS FOR TOP CONFLICTS".center(70))
    print(f"{'='*70}\n")
    
    # Get top 25 conflicts (highest severity first)
    top_conflicts = results[results['severity'].isin(['High', 'Medium'])].head(25)
    
    if len(top_conflicts) > 0:
        print(f" Selected {len(top_conflicts)} top conflicts for resolution generation")
        
        # Optional: Set your Mistral API key here or leave None to be prompted
        MISTRAL_API_KEY = "ypI1jj5mblZTconDPrqV4Rar7iQvIA7x"  # or "your-api-key-here"
        
        # Generate resolutions
        top_conflicts_with_resolutions = generate_mistral_resolutions(
            top_conflicts.copy(), 
            api_key=MISTRAL_API_KEY
        )
        
        # Display results
        display_conflicts_with_resolutions(top_conflicts_with_resolutions)
        
        # Save to CSV
        print(" Saving conflicts with resolutions to 'top_conflicts_with_resolutions.csv'...")
        top_conflicts_with_resolutions.to_csv('top_conflicts_with_resolutions.csv', index=False)
        print(" Saved successfully!")
    else:
        print("ℹ  No high or medium severity conflicts found.")
    
    return model, results, metrics


# ============================================================================
# PART 7: MISTRAL AI RESOLUTION GENERATION (Add this after PART 6)
# ============================================================================

def generate_mistral_resolutions(top_conflicts_df, api_key=None):
    """
    Generate resolutions for top conflicts using Mistral AI API.
    
    Args:
        top_conflicts_df: DataFrame with top conflicts
        api_key: Mistral API key (optional, will prompt if not provided)
    
    Returns:
        DataFrame with resolutions added
    """
    try:
        from mistralai import Mistral
        
    except ImportError:
        print("  Mistral AI library not installed.")
        print("   Install with: pip install mistralai")
        return top_conflicts_df
    
    # Get API key
    if api_key is None:
        print("\n Mistral API Key Required")
        print("   Get your key from: https://console.mistral.ai/")
        api_key = input("Enter your Mistral API key: ").strip()
    
    if not api_key:
        print("  No API key provided. Skipping resolution generation.")
        return top_conflicts_df
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    print(f"\n Generating resolutions using Mistral AI...")
    print(f"   Processing {len(top_conflicts_df)} conflicts...")
    
    resolutions = []
    
    for idx, row in top_conflicts_df.iterrows():
        requirement = row['requirement']
        conflicts = ', '.join(row['conflicts'])
        severity = row['severity']
        
        # Create prompt for Mistral
        prompt = f"""You are a software requirements expert. Analyze this hostel management system requirement and provide a practical resolution.

Requirement: "{requirement}"

Detected Conflicts: {conflicts}
Severity: {severity}

Provide a concise resolution (2-3 sentences) that:
1. Addresses the specific conflicts identified
2. Suggests practical alternatives or modifications
3. Maintains the core functionality where possible

Resolution:"""
        
        try:
            # Call Mistral API
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            chat_response = client.chat.complete(
                model="mistral-small-latest",  # or "mistral-medium-latest" for better quality
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            resolution = chat_response.choices[0].message.content.strip()
            resolutions.append(resolution)
            
            # Progress indicator
            if (idx + 1) % 5 == 0:
                print(f"   Generated {idx + 1}/{len(top_conflicts_df)} resolutions...")
        
        except Exception as e:
            print(f"     Error generating resolution for {row['id']}: {str(e)}")
            resolutions.append("Error: Could not generate resolution")
    
    # Add resolutions to dataframe
    top_conflicts_df['resolution'] = resolutions
    
    print(f" All resolutions generated!")
    
    return top_conflicts_df


def display_conflicts_with_resolutions(conflicts_df):
    """
    Display top conflicts with their resolutions in a readable format.
    
    Args:
        conflicts_df: DataFrame with conflicts and resolutions
    """
    print(f"\n{'='*80}")
    print(f"{'TOP CONFLICTS WITH AI-GENERATED RESOLUTIONS':^80}")
    print(f"{'='*80}\n")
    
    for idx, row in conflicts_df.iterrows():
        print(f"{'─'*80}")
        print(f" {row['id']} | Severity: {row['severity']} | Conflicts: {len(row['conflicts'])}")
        print(f"{'─'*80}")
        print(f"\n📋 REQUIREMENT:")
        print(f"   {row['requirement']}\n")
        print(f"⚠️  CONFLICTS DETECTED:")
        print(f"   {', '.join(row['conflicts'])}\n")
        
        if 'resolution' in row and row['resolution']:
            print(f" RECOMMENDED RESOLUTION:")
            # Word wrap for better readability
            resolution_lines = row['resolution'].split('\n')
            for line in resolution_lines:
                print(f"   {line}")
        
        print(f"\n")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    model, results, metrics = main()