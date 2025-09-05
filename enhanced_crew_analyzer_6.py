import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import warnings
warnings.filterwarnings('ignore')

# CrewAI imports
from crewai import Agent, Task, Crew
from crewai.llm import BaseLLM
from tabulate import tabulate

# Configuration
DATA_DIR = 'realistic_health_data'
OUTPUT_DIR = 'ai_health_reports'
DAILY_DIR = os.path.join(OUTPUT_DIR, 'daily_ai_reports')
WEEKLY_DIR = os.path.join(OUTPUT_DIR, 'weekly_ai_reports')

# Create directories
for directory in [OUTPUT_DIR, DAILY_DIR, WEEKLY_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========================== ENHANCED LLM INTEGRATION ========================== #

class OllamaLLM(BaseLLM):
    """Enhanced Ollama LLM integration with better error handling."""
    
    def __init__(self, model="mistral:latest", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    def call(self, messages, **kwargs):
        """Handle messages properly"""
        try:
            if isinstance(messages, list):
                if all(isinstance(msg, dict) and "content" in msg for msg in messages):
                    prompt = "\n".join([msg["content"] for msg in messages])
                else:
                    prompt = "\n".join(str(msg) for msg in messages)
            else:
                prompt = str(messages)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "top_p": 0.8, "num_predict": 500}
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            return f"LLM Request error: {e}"
        except Exception as err:
            return f"LLM Unexpected error: {err}"

# ========================== PATIENT PROFILE EXTRACTION ========================== #

def extract_patient_profile(personal_info: pd.Series) -> Dict:
    """Extract complete patient profile from Excel data."""
    return {
        'Name': personal_info.get('Name', 'Unknown'),
        'Age': personal_info.get('Age', 'Unknown'),
        'Gender': personal_info.get('Gender', 'Unknown'),
        'Height_cm': personal_info.get('Height', 'Unknown'),
        'Weight_kg': personal_info.get('Weight', 'Unknown'),
        'BMI': personal_info.get('BMI', 'Unknown'),
        'Fitness_Level': personal_info.get('Fitness_Level', 'Unknown'),
        'Diabetes': personal_info.get('Diabetes', 0),
        'Hypertension': personal_info.get('Hypertension', 0),
        'Smoker': personal_info.get('Smoker', 0),
        'Heart_Disease': personal_info.get('Heart_Disease', 0),
        'COPD': personal_info.get('COPD', 0),
        'Arthritis': personal_info.get('Arthritis', 0),
        'Depression': personal_info.get('Depression', 0),
        'Medical History': personal_info.get('Medical History', ''),
        'Current Medications': personal_info.get('Current Medications', '')
    }

def get_time_label(hour: int) -> str:
    """Convert hour to smart time label."""
    if 6 <= hour < 12:
        return f"Morning ({hour:02d}h)"
    elif 12 <= hour < 17:
        return f"Afternoon ({hour:02d}h)"
    elif 17 <= hour < 21:
        return f"Evening ({hour:02d}h)"
    else:
        return f"Night ({hour:02d}h)"

def summarize_patient_profile(profile: Dict) -> str:
    """Create comprehensive patient summary."""
    lines = []
    
    # Demographics
    lines.append("PATIENT PROFILE:")
    lines.append(f"  Name: {profile['Name']}")
    lines.append(f"  Age: {profile['Age']} years, Gender: {profile['Gender']}")
    lines.append(f"  Height: {profile['Height_cm']} cm, Weight: {profile['Weight_kg']} kg")
    lines.append(f"  BMI: {profile['BMI']}, Fitness Level: {profile['Fitness_Level']}")
    
    # Medical Conditions
    conditions = []
    if profile['Diabetes'] == 1:
        conditions.append("Diabetes")
    if profile['Hypertension'] == 1:
        conditions.append("Hypertension")
    if profile['Heart_Disease'] == 1:
        conditions.append("Heart Disease")
    if profile['COPD'] == 1:
        conditions.append("COPD")
    if profile['Arthritis'] == 1:
        conditions.append("Arthritis")
    if profile['Depression'] == 1:
        conditions.append("Depression")
    
    if conditions:
        lines.append(f"  Medical Conditions: {', '.join(conditions)}")
    else:
        lines.append("  Medical Conditions: None")
    
    # Risk Factors
    risk_factors = []
    if profile['Smoker'] == 1:
        risk_factors.append("Smoker")
    
    bmi = profile['BMI']
    if isinstance(bmi, (int, float)):
        if bmi >= 40:
            risk_factors.append("Severely Obese (BMI ≥40)")
        elif bmi >= 30:
            risk_factors.append("Obese (BMI ≥30)")
    
    if risk_factors:
        lines.append(f"  Risk Factors: {', '.join(risk_factors)}")
    
    # Clinical Targets
    lines.append("  Clinical Targets:")
    if profile['Diabetes'] == 1:
        lines.append("    - Glucose: 80-130 mg/dL fasting, <180 mg/dL post-meal")
    if profile['Hypertension'] == 1:
        lines.append("    - Blood Pressure: <130/80 mmHg")
    if profile['COPD'] == 1:
        lines.append("    - SpO2: 88-92% (COPD target)")
    else:
        lines.append("    - SpO2: 95-100%")
    lines.append("    - Heart Rate: 60-100 bpm")
    
    return "\n".join(lines)

# ========================== SMART HEALTH DATA ANALYSIS ========================== #

def analyze_hourly_patterns(data: pd.DataFrame, value_col: str, threshold_low: float = None, threshold_high: float = None) -> str:
    """Analyze hourly patterns and return smart summary."""
    if data.empty:
        return "No data available"
    
    data['Hour'] = data['Timestamp'].dt.hour
    hourly_stats = data.groupby('Hour')[value_col].agg(['mean', 'count']).round(1)
    
    issues = []
    
    # Find problematic hours
    if threshold_low is not None:
        low_hours = hourly_stats[hourly_stats['mean'] < threshold_low].index.tolist()
        if low_hours:
            time_periods = [get_time_label(h) for h in low_hours[:3]]
            avg_val = hourly_stats.loc[low_hours, 'mean'].mean()
            issues.append(f"Low values: {', '.join(time_periods)} (avg {avg_val:.1f})")
    
    if threshold_high is not None:
        high_hours = hourly_stats[hourly_stats['mean'] > threshold_high].index.tolist()
        if high_hours:
            time_periods = [get_time_label(h) for h in high_hours[:3]]
            avg_val = hourly_stats.loc[high_hours, 'mean'].mean()
            issues.append(f"High values: {', '.join(time_periods)} (avg {avg_val:.1f})")
    
    return " | ".join(issues) if issues else "Stable throughout day"

def create_daily_health_summary(user_data: Dict, date: datetime.date, patient_profile: Dict) -> str:
    """Create comprehensive but concise daily health summary."""
    summary = []
    
    # Header
    summary.append(f"HEALTH ANALYSIS - {patient_profile['Name']} - {date}")
    summary.append("=" * 60)
    
    # Patient Overview
    age = patient_profile['Age']
    gender = patient_profile['Gender']
    bmi = patient_profile['BMI']
    conditions = []
    if patient_profile['Diabetes'] == 1:
        conditions.append("Diabetes")
    if patient_profile['Hypertension'] == 1:
        conditions.append("Hypertension")
    if patient_profile['COPD'] == 1:
        conditions.append("COPD")
    if patient_profile['Heart_Disease'] == 1:
        conditions.append("Heart Disease")
    
    summary.append(f"Patient: {age}y {gender}, BMI {bmi}")
    if conditions:
        summary.append(f"Conditions: {', '.join(conditions)}")
    
    # Sleep Analysis
    if 'Daily Sleep' in user_data:
        sleep_data = user_data['Daily Sleep'][user_data['Daily Sleep']['Date'].dt.date == date]
        if not sleep_data.empty:
            sleep_row = sleep_data.iloc[0]
            total_hours = sleep_row['Sleep Time'] / 60
            deep_pct = (sleep_row['Deep Sleep'] / sleep_row['Sleep Time']) * 100
            rem_pct = (sleep_row['REM Sleep'] / sleep_row['Sleep Time']) * 100
            
            summary.append(f"\nSLEEP: {total_hours:.1f}h total")
            summary.append(f"  Deep: {deep_pct:.1f}% (target 15-25%), REM: {rem_pct:.1f}% (target 20-25%)")
            
            issues = []
            if total_hours < 6:
                issues.append("Sleep deprived (<6h)")
            if deep_pct < 15:
                issues.append("Low deep sleep")
            if rem_pct < 20:
                issues.append("Low REM sleep")
            
            if issues:
                summary.append(f"  Issues: {' | '.join(issues)}")
            else:
                summary.append("  Quality: Good sleep architecture")
    
    # Activity Analysis
    if 'Daily Steps' in user_data:
        activity_data = user_data['Daily Steps'][user_data['Daily Steps']['Date'].dt.date == date]
        if not activity_data.empty:
            activity_row = activity_data.iloc[0]
            steps = int(activity_row['Step Count'])
            
            if steps < 3000:
                level = "Sedentary"
            elif steps < 5000:
                level = "Low"
            elif steps < 7500:
                level = "Moderate"
            elif steps < 10000:
                level = "Active"
            else:
                level = "Highly Active"
            
            summary.append(f"\nACTIVITY: {steps:,} steps ({level})")
            summary.append(f"  Calories: {activity_row['Calorie']:.0f} kcal, Distance: {activity_row['Mileage']:.1f} miles")
    
    # HRV Analysis
    if 'Segmented HRV' in user_data:
        hrv_data = user_data['Segmented HRV'][user_data['Segmented HRV']['Timestamp'].dt.date == date]
        if not hrv_data.empty:
            avg_hrv = hrv_data['HRV'].mean()
            min_hrv = hrv_data['HRV'].min()
            max_hrv = hrv_data['HRV'].max()
            
            # Age-adjusted targets
            age = patient_profile['Age']
            if isinstance(age, (int, float)):
                if age >= 60:
                    target_hrv = 25
                elif age >= 40:
                    target_hrv = 35
                else:
                    target_hrv = 45
            else:
                target_hrv = 30
            
            summary.append(f"\nHRV: {avg_hrv:.1f}ms avg (target >{target_hrv}ms), range {min_hrv:.1f}-{max_hrv:.1f}ms")
            
            # Distribution analysis
            very_low = hrv_data[hrv_data['HRV'] < 20]
            low = hrv_data[(hrv_data['HRV'] >= 20) & (hrv_data['HRV'] < 40)]
            normal = hrv_data[hrv_data['HRV'] >= 40]
            
            dist_parts = []
            if len(very_low) > 0:
                pct = len(very_low) / len(hrv_data) * 100
                if pct > 50:
                    pattern = analyze_hourly_patterns(very_low, 'HRV', threshold_high=19.9)
                    dist_parts.append(f"Very low ({pct:.0f}%): {pattern}")
                else:
                    dist_parts.append(f"Very low: {pct:.0f}%")
            
            if len(normal) > 0:
                pct = len(normal) / len(hrv_data) * 100
                dist_parts.append(f"Normal: {pct:.0f}%")
            
            if dist_parts:
                summary.append(f"  Distribution: {' | '.join(dist_parts)}")
    
    # SpO2 Analysis
    if 'Segmented SPO2' in user_data:
        spo2_data = user_data['Segmented SPO2'][user_data['Segmented SPO2']['Timestamp'].dt.date == date]
        if not spo2_data.empty:
            avg_spo2 = spo2_data['SPO2'].mean()
            min_spo2 = spo2_data['SPO2'].min()
            
            has_copd = patient_profile['COPD'] == 1
            target_min = 88 if has_copd else 95
            
            summary.append(f"\nSpO2: {avg_spo2:.1f}% avg, {min_spo2:.1f}% min (target >{target_min}%)")
            
            concerning = spo2_data[spo2_data['SPO2'] < target_min]
            if len(concerning) > 0:
                pct = len(concerning) / len(spo2_data) * 100
                pattern = analyze_hourly_patterns(concerning, 'SPO2', threshold_low=target_min)
                summary.append(f"  Concerning: {pct:.0f}% of readings - {pattern}")
            else:
                summary.append("  Status: All readings within target range")
    
    # Vital Signs Analysis
    if 'Segmented Health Signals' in user_data:
        vitals_data = user_data['Segmented Health Signals'][user_data['Segmented Health Signals']['Timestamp'].dt.date == date]
        if not vitals_data.empty:
            avg_hr = vitals_data['Heart Rate'].mean()
            avg_sys = vitals_data['Systolic BP'].mean()
            avg_dia = vitals_data['Diastolic BP'].mean()
            avg_glucose = vitals_data['Blood Glucose'].mean()
            glucose_range = f"{vitals_data['Blood Glucose'].min():.0f}-{vitals_data['Blood Glucose'].max():.0f}"
            
            summary.append(f"\nVITALS: HR {avg_hr:.0f}bpm, BP {avg_sys:.0f}/{avg_dia:.0f}mmHg, Glucose {avg_glucose:.0f}mg/dL ({glucose_range})")
            
            # Clinical interpretation
            clinical_notes = []
            
            # Heart Rate
            if avg_hr > 100:
                hr_high = vitals_data[vitals_data['Heart Rate'] > 100]
                pattern = analyze_hourly_patterns(hr_high, 'Heart Rate', threshold_low=100)
                clinical_notes.append(f"Tachycardia: {pattern}")
            elif avg_hr < 60 and patient_profile['Fitness_Level'] not in ['excellent', 'very good']:
                clinical_notes.append(f"Bradycardia: {avg_hr:.0f}bpm")
            
            # Blood Pressure
            has_hypertension = patient_profile['Hypertension'] == 1
            bp_target_sys = 130 if has_hypertension else 120
            bp_target_dia = 80
            
            if avg_sys > bp_target_sys or avg_dia > bp_target_dia:
                bp_high = vitals_data[(vitals_data['Systolic BP'] > bp_target_sys) | (vitals_data['Diastolic BP'] > bp_target_dia)]
                pattern = analyze_hourly_patterns(bp_high, 'Systolic BP', threshold_low=bp_target_sys)
                status = "Uncontrolled HTN" if has_hypertension else "Elevated BP"
                clinical_notes.append(f"{status}: {pattern}")
            
            # Glucose
            has_diabetes = patient_profile['Diabetes'] == 1
            glucose_target = 180 if has_diabetes else 140
            
            glucose_high = vitals_data[vitals_data['Blood Glucose'] > glucose_target]
            glucose_low = vitals_data[vitals_data['Blood Glucose'] < 70]
            
            if len(glucose_high) > 0:
                pattern = analyze_hourly_patterns(glucose_high, 'Blood Glucose', threshold_low=glucose_target)
                status = "Hyperglycemia" if has_diabetes else "High glucose"
                clinical_notes.append(f"{status}: {pattern}")
            
            if len(glucose_low) > 0:
                pattern = analyze_hourly_patterns(glucose_low, 'Blood Glucose', threshold_high=69)
                clinical_notes.append(f"Hypoglycemia: {pattern}")
            
            # Display clinical findings
            if clinical_notes:
                summary.append(f"  Clinical Findings: {' | '.join(clinical_notes)}")
            else:
                summary.append("  Status: All vitals within target ranges")
            
            # Reference ranges
            ref_ranges = []
            ref_ranges.append("HR 60-100bpm")
            if has_hypertension:
                ref_ranges.append("BP <130/80mmHg")
            if has_diabetes:
                ref_ranges.append("Glucose <180mg/dL")
            
            summary.append(f"  Targets: {' | '.join(ref_ranges)}")
    
    # Sleep Architecture
    if 'Segmented Sleep' in user_data:
        sleep_stages = user_data['Segmented Sleep'][user_data['Segmented Sleep']['Timestamp'].dt.date == date]
        if not sleep_stages.empty:
            stage_counts = sleep_stages['Sleep Stage'].value_counts()
            total_segments = len(sleep_stages)
            
            stage_summary = []
            issues = []
            
            for stage in ['Deep', 'REM', 'Light', 'Awake']:
                if stage in stage_counts:
                    pct = (stage_counts[stage] / total_segments) * 100
                    stage_summary.append(f"{stage} {pct:.0f}%")
                    
                    if stage == 'Deep' and pct < 15:
                        issues.append("Low deep sleep")
                    elif stage == 'REM' and pct < 20:
                        issues.append("Low REM")
                    elif stage == 'Awake' and pct > 10:
                        issues.append("Frequent awakenings")
            
            summary.append(f"\nSLEEP STAGES: {' | '.join(stage_summary)}")
            if issues:
                summary.append(f"  Issues: {' | '.join(issues)}")
            else:
                summary.append("  Quality: Good sleep architecture")
    
    return "\n".join(summary)

# ========================== AI AGENTS ========================== #

def create_health_agents(llm_instance):
    """Create specialized health analysis agents."""
    
    summary_agent = Agent(
        role="Clinical Data Analyst",
        goal="Provide patient-specific clinical interpretation of health data",
        backstory=(
            "You are an experienced clinical analyst specializing in chronic disease management. "
            "You interpret health data considering patient's specific conditions (diabetes, hypertension, "
            "COPD, etc.), age, fitness level, and medical history to provide relevant clinical insights."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    clinical_agent = Agent(
        role="Clinical Assessment Specialist",
        goal="Provide additional clinical context and detailed health information",
        backstory=(
            "You are a clinical specialist who provides comprehensive clinical context rather than "
            "just identifying abnormalities. You explain what health patterns mean for patients with "
            "specific conditions and provide detailed clinical information for better understanding."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    advisor_agent = Agent(
        role="Personalized Health Coach",
        goal="Provide evidence-based, personalized health recommendations",
        backstory=(
            "You are a health coach specializing in chronic disease management. You provide realistic, "
            "achievable health recommendations based on patient's conditions, limitations, and current "
            "health status. Your advice considers safety and medical appropriateness."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    synthesizer_agent = Agent(
        role="Chief Medical Synthesizer",
        goal="Integrate all analyses into comprehensive, actionable health insights",
        backstory=(
            "You are a senior clinician who synthesizes complex health data into clear, prioritized "
            "insights for patients with chronic conditions. You balance medical accuracy with practical "
            "guidance, always considering patient safety and realistic health goals."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    trend_agent = Agent(
        role="Health Trend Analyst",
        goal="Analyze weekly patterns and chronic disease management trends",
        backstory=(
            "You are a health analytics specialist focused on chronic disease management trends. "
            "You identify patterns in weekly data, assess disease control effectiveness, and provide "
            "recommendations for optimizing long-term health management."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    return {
        'summary': summary_agent,
        'clinical': clinical_agent,
        'advisor': advisor_agent,
        'synthesizer': synthesizer_agent,
        'trend': trend_agent
    }

# ========================== TASK CREATION ========================== #

def create_analysis_tasks(date: datetime.date, health_summary: str, agents: Dict, patient_info: str):
    """Create analysis tasks for the AI agents."""
    
    summary_task = Task(
        description=(
            f"PATIENT PROFILE:\n{patient_info}\n\n"
            f"Analyze this comprehensive health data for {date} considering the patient's specific "
            f"medical conditions, age, and health profile:\n\n{health_summary}\n\n"
            f"Provide clinical interpretation focusing on chronic disease management, sleep quality, "
            f"cardiovascular health, respiratory function, and activity levels appropriate for this patient."
        ),
        expected_output="A clinical interpretation (3-4 sentences) highlighting key findings and their significance for this patient's health management.",
        agent=agents['summary']
    )
    
    clinical_task = Task(
        description=(
            f"PATIENT PROFILE:\n{patient_info}\n\n"
            f"Provide detailed clinical context for the health data from {date}:\n\n{health_summary}\n\n"
            f"Focus on explaining what the findings mean clinically rather than just identifying abnormal values. "
            f"Consider disease interactions, optimal management strategies, and provide additional diagnostic insights."
        ),
        expected_output="Detailed clinical assessment (3-4 sentences) providing context and additional information about the patient's health status.",
        agent=agents['clinical']
    )
    
    advisor_task = Task(
        description=(
            f"PATIENT PROFILE:\n{patient_info}\n\n"
            f"Based on the health data for {date}, provide personalized health recommendations:\n\n{health_summary}\n\n"
            f"Consider the patient's medical conditions, fitness level, age, and limitations. Provide realistic, "
            f"evidence-based recommendations that are safe and appropriate for their health status."
        ),
        expected_output="Two specific, actionable health recommendations (2-3 sentences) appropriate for this patient's conditions and capabilities.",
        agent=agents['advisor']
    )
    
    final_task = Task(
        description=(
            f"PATIENT PROFILE:\n{patient_info}\n\n"
            f"Synthesize all analyses into a comprehensive daily health summary for {date}. "
            f"Integrate clinical interpretation, additional clinical information, and recommendations "
            f"into clear, prioritized insights for chronic disease management."
        ),
        expected_output="A comprehensive daily health summary (5-6 sentences) integrating all analyses into actionable insights for the patient and healthcare team.",
        agent=agents['synthesizer'],
        context=[summary_task, clinical_task, advisor_task]
    )
    
    return [summary_task, clinical_task, advisor_task, final_task]

def create_weekly_task(weekly_data: str, agent: Agent, patient_info: str):
    """Create weekly trend analysis task."""
    return Task(
        description=(
            f"PATIENT PROFILE:\n{patient_info}\n\n"
            f"Analyze the following 7 days of health summaries for patterns and trends:\n\n{weekly_data}\n\n"
            f"Focus on chronic disease management effectiveness, sleep patterns, activity consistency, "
            f"vital signs trends, and overall health trajectory. Identify improvements and areas needing attention."
        ),
        expected_output="A comprehensive weekly report (6-8 sentences) identifying trends, improvements, concerns, and specific recommendations for continued health optimization.",
        agent=agent
    )

# ========================== MAIN PROCESSING ========================== #

def process_user_with_ai_agents(file_path: str, llm_instance) -> None:
    """Process user data with AI agent analysis."""
    try:
        filename = os.path.basename(file_path)
        user_id = filename.split('_')[1]
        
        print(f"🤖 Processing: {filename}")
        
        # Load user data
        user_data = pd.read_excel(file_path, sheet_name=None)
        
        # Check if Personal Info exists
        if 'Personal Info' not in user_data or user_data['Personal Info'].empty:
            print(f"⚠️  No personal info found for user {user_id}, skipping...")
            return
            
        # Extract patient profile
        personal_info_row = user_data['Personal Info'].iloc[0]
        patient_profile = extract_patient_profile(personal_info_row)
        patient_summary = summarize_patient_profile(patient_profile)
        
        print(f"   👤 Patient: {patient_profile['Name']}, {patient_profile['Age']}y, {patient_profile['Gender']}")
        
        # Display conditions
        conditions = []
        for condition in ['Diabetes', 'Hypertension', 'Heart_Disease', 'COPD', 'Depression', 'Arthritis']:
            if patient_profile[condition] == 1:
                conditions.append(condition.replace('_', ' '))
        
        if conditions:
            print(f"   🏥 Conditions: {', '.join(conditions)}")
        
        if patient_profile['Smoker'] == 1:
            print(f"   ⚠️  Risk: Smoker")
        
        # Convert date columns
        for sheet_name in user_data:
            if 'Date' in user_data[sheet_name].columns:
                user_data[sheet_name]['Date'] = pd.to_datetime(user_data[sheet_name]['Date'])
            if 'Timestamp' in user_data[sheet_name].columns:
                user_data[sheet_name]['Timestamp'] = pd.to_datetime(user_data[sheet_name]['Timestamp'])
        
        # Create AI agents
        agents = create_health_agents(llm_instance)
        
        # Get available dates
        dates = user_data['Daily Steps']['Date'].dt.date.unique() if 'Daily Steps' in user_data else []
        daily_summaries = []
        
        # Process each day
        for date in sorted(dates):
            print(f"  🧠 Analyzing: {date}")
            
            # Create health summary
            health_summary = create_daily_health_summary(user_data, date, patient_profile)
            
            # Create and execute tasks
            tasks = create_analysis_tasks(date, health_summary, agents, patient_summary)
            
            crew = Crew(
                agents=[agents['summary'], agents['clinical'], agents['advisor'], agents['synthesizer']],
                tasks=tasks,
                verbose=False
            )
            
            try:
                result = crew.kickoff()
                ai_summary = result.raw if hasattr(result, 'raw') else str(result)
                
                # Save daily analysis
                daily_filename = f"user_{user_id}_day_{date}_analysis.txt"
                daily_path = os.path.join(DAILY_DIR, daily_filename)
                
                full_report = f"AI HEALTH ANALYSIS - {patient_profile['Name']} - {date}\n"
                full_report += "=" * 60 + "\n\n"
                full_report += f"{patient_summary}\n\n"
                full_report += f"HEALTH DATA SUMMARY:\n{health_summary}\n\n"
                full_report += f"AI ANALYSIS:\n{ai_summary}\n"
                
                with open(daily_path, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                
                daily_summaries.append((date, ai_summary))
                
            except Exception as e:
                print(f"    ⚠️  Error in analysis for {date}: {str(e)}")
                continue
        
        # Generate weekly analysis
        if len(daily_summaries) >= 7:
            print(f"  📈 Weekly trend analysis for User {user_id}")
            
            last_7_days = daily_summaries[-7:]
            weekly_input = "\n\n".join([f"Date: {date}\nAnalysis: {summary}" for date, summary in last_7_days])
            
            weekly_task = create_weekly_task(weekly_input, agents['trend'], patient_summary)
            weekly_crew = Crew(agents=[agents['trend']], tasks=[weekly_task], verbose=False)
            
            try:
                weekly_result = weekly_crew.kickoff()
                weekly_summary = weekly_result.raw if hasattr(weekly_result, 'raw') else str(weekly_result)
                
                # Save weekly analysis
                week_start = last_7_days[0][0]
                weekly_filename = f"user_{user_id}_week_{week_start}_trends.txt"
                weekly_path = os.path.join(WEEKLY_DIR, weekly_filename)
                
                weekly_report = f"AI WEEKLY TREND ANALYSIS - {patient_profile['Name']}\n"
                weekly_report += f"Week: {week_start} to {last_7_days[-1][0]}\n"
                weekly_report += "=" * 60 + "\n\n"
                weekly_report += f"{patient_summary}\n\n"
                weekly_report += f"WEEKLY TRENDS:\n{weekly_summary}\n\n"
                weekly_report += "DAILY SUMMARIES:\n" + "-" * 40 + "\n"
                for date, summary in last_7_days:
                    weekly_report += f"\n{date}:\n{summary}\n" + "-" * 40 + "\n"
                
                with open(weekly_path, 'w', encoding='utf-8') as f:
                    f.write(weekly_report)
                
                print(f"    ✅ Weekly analysis saved: {weekly_filename}")
                
            except Exception as e:
                print(f"    ⚠️  Error in weekly analysis: {str(e)}")
        
        print(f"✅ Completed analysis for User {user_id}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""
    print("🏥 SMART AI HEALTH DATA ANALYZER")
    print("=" * 50)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🤖 AI reports directory: {OUTPUT_DIR}")
    print("=" * 50)
    
    # Initialize LLM
    try:
        llm = OllamaLLM(model="mistral:latest")
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    
    # Check data directory
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found!")
        return
    
    # Find Excel files
    excel_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]
    if not excel_files:
        print(f"❌ No Excel files found in '{DATA_DIR}'!")
        return
    
    print(f"📊 Found {len(excel_files)} user data files")
    print("🚀 Starting smart AI health analysis...")
    
    # Process each file
    processed_count = 0
    for filename in sorted(excel_files):
        file_path = os.path.join(DATA_DIR, filename)
        process_user_with_ai_agents(file_path, llm)
        processed_count += 1
    
    print("\n" + "=" * 50)
    print(f"🎉 ANALYSIS COMPLETE!")
    print(f"✅ Processed {processed_count} users")
    
    # Summary statistics
    daily_files = len([f for f in os.listdir(DAILY_DIR) if f.endswith('.txt')])
    weekly_files = len([f for f in os.listdir(WEEKLY_DIR) if f.endswith('.txt')])
    
    print(f"\n📊 RESULTS:")
    print(f"   • Daily analyses: {daily_files}")
    print(f"   • Weekly trend reports: {weekly_files}")
    if processed_count > 0:
        print(f"   • Average daily reports per user: {daily_files/processed_count:.1f}")
    
    print(f"\n🧠 KEY FEATURES:")
    print(f"   ✓ Complete patient profile extraction")
    print(f"   ✓ All medical conditions included")
    print(f"   ✓ Smart time-based pattern analysis")
    print(f"   ✓ Clinical context with detailed information")
    print(f"   ✓ Patient-specific targets and recommendations")
    print(f"   ✓ Chronic disease management focus")
    print(f"   ✓ Weekly trend analysis")
    print(f"   ✓ Simplified, clear, and actionable insights")
    
    print("=" * 50)

if __name__ == "__main__":
    main()