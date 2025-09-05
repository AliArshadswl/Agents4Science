#!/usr/bin/env python3
"""
Enhanced CrewAI Health Data Analyzer
====================================

Multi-agent system for comprehensive health data analysis using CrewAI.
Integrates with the medically realistic health dataset for advanced insights.

Agents:
- Summary Agent: Clinical interpretation of daily metrics
- Anomaly Agent: Risk assessment and pattern detection  
- Suggestion Agent: Personalized health recommendations
- Trend Agent: Weekly pattern analysis
- Final Agent: Comprehensive report synthesis

Author: Health AI Analytics Team
Version: 2.0 - Enhanced for Comprehensive Health Data
"""

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
ANOMALY_DIR = os.path.join(OUTPUT_DIR, 'anomaly_logs')

# Create directories
for directory in [OUTPUT_DIR, DAILY_DIR, WEEKLY_DIR, ANOMALY_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========================== ENHANCED LLM INTEGRATION ========================== #

class OllamaLLM(BaseLLM):
    """Enhanced Ollama LLM integration with better error handling and context management."""
    
    def __init__(self, model="mistral:latest", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    def call(self, prompt, **kwargs):
        try:
            # Handle different prompt formats
            if isinstance(prompt, list):
                if all(isinstance(p, dict) and "content" in p for p in prompt):
                    prompt = "\n".join([p["content"] for p in prompt])
                else:
                    prompt = "\n".join(str(p) for p in prompt)
            
            # Ensure prompt is string
            prompt = str(prompt)
            
            # Make API call to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent medical analysis
                        "top_p": 0.8,
                        "num_predict": 500   # Limit response length
                    }
                },
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            return f"LLM Request error: {e}"
        except Exception as err:
            return f"LLM Unexpected error: {err}"

# ========================== ENHANCED DATA PROCESSING FUNCTIONS ========================== #

def summarize_personal_info_enhanced(info: pd.Series) -> str:
    """Enhanced personal information summary with comprehensive health profile."""
    smoker = 'does not smoke' if info.get('Smoker', 0) == 0 else 'smokes'
    
    # Comprehensive condition mapping
    conditions = []
    condition_mapping = {
        'Diabetes': 'type 2 diabetes',
        'Hypertension': 'hypertension',
        'Heart_Disease': 'coronary heart disease',
        'COPD': 'chronic obstructive pulmonary disease',
        'Arthritis': 'arthritis',
        'Depression': 'depression'
    }
    
    for condition, label in condition_mapping.items():
        if condition in info and info[condition] == 1:
            conditions.append(label)
    
    condition_summary = "has no known chronic conditions" if not conditions else "has " + ", ".join(conditions)
    
    # BMI and fitness assessment
    bmi = info.get('BMI', 0)
    fitness_level = info.get('Fitness_Level', 'unknown')
    
    # Risk stratification
    risk_factors = []
    if info.get('Age', 0) >= 65:
        risk_factors.append("advanced age")
    if bmi >= 30:
        risk_factors.append("obesity")
    if info.get('Smoker', 0) == 1:
        risk_factors.append("tobacco use")
    
    risk_summary = f" Risk factors: {', '.join(risk_factors)}." if risk_factors else ""
    
    return (
        f"Patient: {info.get('Name', 'Unknown')}, {info.get('Age', 0)} years old "
        f"{info.get('Gender', 'unknown').lower()}, BMI {bmi:.1f}, "
        f"fitness level: {fitness_level}. Medical history: {condition_summary}. "
        f"Smoking status: {smoker}.{risk_summary}"
    )

def create_comprehensive_daily_summary(user_data: Dict, date: datetime.date) -> str:
    """Create comprehensive daily health summary for AI analysis."""
    summary_sections = []
    
    # Date and basic info
    summary_sections.append(f"HEALTH DATA SUMMARY FOR {date}")
    summary_sections.append("=" * 50)
    
    # Sleep Analysis
    if 'Daily Sleep' in user_data:
        sleep_data = user_data['Daily Sleep'][user_data['Daily Sleep']['Date'].dt.date == date]
        if not sleep_data.empty:
            sleep_row = sleep_data.iloc[0]
            total_sleep_hours = sleep_row['Sleep Time'] / 60
            deep_sleep_pct = (sleep_row['Deep Sleep'] / sleep_row['Sleep Time']) * 100
            rem_sleep_pct = (sleep_row['REM Sleep'] / sleep_row['Sleep Time']) * 100
            awake_pct = (sleep_row['Awake Time'] / sleep_row['Sleep Time']) * 100
            
            summary_sections.append(f"\nSLEEP METRICS:")
            summary_sections.append(f"  Total sleep: {total_sleep_hours:.1f} hours")
            summary_sections.append(f"  Deep sleep: {sleep_row['Deep Sleep']:.0f}min ({deep_sleep_pct:.1f}%)")
            summary_sections.append(f"  REM sleep: {sleep_row['REM Sleep']:.0f}min ({rem_sleep_pct:.1f}%)")
            summary_sections.append(f"  Wake time: {sleep_row['Awake Time']:.0f}min ({awake_pct:.1f}%)")
    
    # Activity Analysis
    if 'Daily Steps' in user_data:
        activity_data = user_data['Daily Steps'][user_data['Daily Steps']['Date'].dt.date == date]
        if not activity_data.empty:
            activity_row = activity_data.iloc[0]
            summary_sections.append(f"\nPHYSICAL ACTIVITY:")
            summary_sections.append(f"  Steps: {int(activity_row['Step Count']):,}")
            summary_sections.append(f"  Calories burned: {activity_row['Calorie']:.0f} kcal")
            summary_sections.append(f"  Distance: {activity_row['Mileage']:.2f} miles")
    
    # Cardiovascular Metrics
    if 'Segmented HRV' in user_data:
        hrv_data = user_data['Segmented HRV'][user_data['Segmented HRV']['Timestamp'].dt.date == date]
        if not hrv_data.empty:
            avg_hrv = hrv_data['HRV'].mean()
            min_hrv = hrv_data['HRV'].min()
            max_hrv = hrv_data['HRV'].max()
            std_hrv = hrv_data['HRV'].std()
            low_hrv_count = len(hrv_data[hrv_data['HRV'] < 25])
            
            summary_sections.append(f"\nHEART RATE VARIABILITY:")
            summary_sections.append(f"  Average HRV: {avg_hrv:.1f}ms")
            summary_sections.append(f"  Range: {min_hrv:.1f} - {max_hrv:.1f}ms")
            summary_sections.append(f"  Variability: ±{std_hrv:.1f}ms")
            if low_hrv_count > 0:
                summary_sections.append(f"  Low HRV episodes (<25ms): {low_hrv_count}")
    
    # Respiratory Metrics
    if 'Segmented SPO2' in user_data:
        spo2_data = user_data['Segmented SPO2'][user_data['Segmented SPO2']['Timestamp'].dt.date == date]
        if not spo2_data.empty:
            avg_spo2 = spo2_data['SPO2'].mean()
            min_spo2 = spo2_data['SPO2'].min()
            hypoxemia_count = len(spo2_data[spo2_data['SPO2'] < 90])
            severe_hypoxemia_count = len(spo2_data[spo2_data['SPO2'] < 88])
            
            summary_sections.append(f"\nOXYGEN SATURATION:")
            summary_sections.append(f"  Average SpO2: {avg_spo2:.1f}%")
            summary_sections.append(f"  Minimum SpO2: {min_spo2:.1f}%")
            if hypoxemia_count > 0:
                summary_sections.append(f"  Hypoxemia episodes (<90%): {hypoxemia_count}")
            if severe_hypoxemia_count > 0:
                summary_sections.append(f"  Severe hypoxemia (<88%): {severe_hypoxemia_count}")
    
    # Vital Signs Summary
    if 'Segmented Health Signals' in user_data:
        vitals_data = user_data['Segmented Health Signals'][user_data['Segmented Health Signals']['Timestamp'].dt.date == date]
        if not vitals_data.empty:
            avg_hr = vitals_data['Heart Rate'].mean()
            avg_systolic = vitals_data['Systolic BP'].mean()
            avg_diastolic = vitals_data['Diastolic BP'].mean()
            avg_glucose = vitals_data['Blood Glucose'].mean()
            max_glucose = vitals_data['Blood Glucose'].max()
            high_glucose_count = len(vitals_data[vitals_data['Blood Glucose'] > 200])
            
            summary_sections.append(f"\nVITAL SIGNS:")
            summary_sections.append(f"  Heart rate: {avg_hr:.0f} bpm")
            summary_sections.append(f"  Blood pressure: {avg_systolic:.0f}/{avg_diastolic:.0f} mmHg")
            summary_sections.append(f"  Blood glucose: {avg_glucose:.0f} mg/dL (max: {max_glucose:.0f})")
            if high_glucose_count > 0:
                summary_sections.append(f"  Hyperglycemia episodes (>200mg/dL): {high_glucose_count}")
    
    # Sleep Architecture
    if 'Segmented Sleep' in user_data:
        sleep_stages = user_data['Segmented Sleep'][user_data['Segmented Sleep']['Timestamp'].dt.date == date]
        if not sleep_stages.empty:
            stage_counts = sleep_stages['Sleep Stage'].value_counts()
            total_segments = len(sleep_stages)
            
            summary_sections.append(f"\nSLEEP ARCHITECTURE:")
            for stage in ['Light', 'Deep', 'REM', 'Awake']:
                count = stage_counts.get(stage, 0)
                percentage = (count / total_segments) * 100
                summary_sections.append(f"  {stage}: {count} periods ({percentage:.1f}%)")
    
    return "\n".join(summary_sections)

# ========================== ENHANCED AGENT DEFINITIONS ========================== #

def create_health_agents(llm_instance):
    """Create specialized health analysis agents with enhanced capabilities."""
    
    summary_agent = Agent(
        role="Clinical Data Interpreter",
        goal="Provide accurate, contextual interpretation of daily health metrics",
        backstory=(
            "You are an experienced clinical data analyst specializing in wearable health technology. "
            "You have extensive experience interpreting physiological data from continuous monitoring devices "
            "and understand how to contextualize findings within individual patient profiles including "
            "age, comorbidities, and risk factors. You provide clear, medically accurate summaries."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    anomaly_agent = Agent(
        role="Clinical Risk Assessor",
        goal="Identify significant health anomalies and assess clinical relevance",
        backstory=(
            "You are a clinical specialist focused on risk assessment and anomaly detection in health monitoring data. "
            "You excel at distinguishing between normal variations and clinically significant deviations, "
            "always considering the patient's individual health profile, medical history, and risk factors. "
            "You provide evidence-based assessments of health patterns and potential concerns."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    suggestion_agent = Agent(
        role="Personalized Health Advisor",
        goal="Provide tailored, actionable health recommendations",
        backstory=(
            "You are a wellness coach and health educator with deep knowledge of lifestyle medicine. "
            "You specialize in creating personalized, achievable health recommendations based on individual "
            "health data, medical conditions, and lifestyle factors. Your advice is always medically appropriate, "
            "realistic, and considers the patient's current health status and limitations."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    trend_agent = Agent(
        role="Health Trend Analyst",
        goal="Analyze weekly patterns and health trends",
        backstory=(
            "You are a health analytics specialist who excels at identifying patterns and trends in longitudinal "
            "health data. You understand how to track progress, identify concerning trends, and recognize "
            "improvements over time. You provide insights into weekly health patterns and their clinical significance."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    final_agent = Agent(
        role="Chief Medical Summarizer",
        goal="Synthesize all analyses into comprehensive, actionable health insights",
        backstory=(
            "You are a senior clinician who specializes in synthesizing complex health data into clear, "
            "actionable insights for patients and healthcare providers. You excel at integrating multiple "
            "data streams and expert analyses to provide comprehensive, personalized health guidance that "
            "prioritizes patient safety and well-being."
        ),
        llm=llm_instance,
        verbose=True
    )
    
    return {
        'summary': summary_agent,
        'anomaly': anomaly_agent,
        'suggestion': suggestion_agent,
        'trend': trend_agent,
        'final': final_agent
    }

# ========================== ENHANCED TASK CREATION FUNCTIONS ========================== #

def create_enhanced_summary_task(date: datetime.date, health_summary: str, agent: Agent, personal_info: str) -> Task:
    """Create enhanced summary task with comprehensive health context."""
    return Task(
        description=(
            f"PATIENT PROFILE:\n{personal_info}\n\n"
            f"CLINICAL TASK: Analyze the following comprehensive health monitoring data for {date}. "
            f"Provide a clinical interpretation of the physiological parameters, considering the patient's "
            f"medical history, risk factors, and normal variations for their profile.\n\n"
            f"HEALTH DATA:\n{health_summary}\n\n"
            f"Focus on: sleep quality and architecture, cardiovascular function (HRV), respiratory status (SpO2), "
            f"physical activity levels, vital signs trends, and any metabolic indicators."
        ),
        expected_output=(
            "A clinical paragraph (3-4 sentences) interpreting the key physiological findings, "
            "noting any significant patterns or values that warrant attention given the patient's profile."
        ),
        agent=agent
    )

def create_enhanced_anomaly_task(date: datetime.date, health_summary: str, agent: Agent, personal_info: str) -> Task:
    """Create enhanced anomaly detection task with clinical context."""
    return Task(
        description=(
            f"PATIENT PROFILE:\n{personal_info}\n\n"
            f"CLINICAL RISK ASSESSMENT: Carefully review the health data for {date} to identify any "
            f"significant anomalies or concerning patterns. Consider the patient's medical history, "
            f"current conditions, and risk factors when determining clinical significance.\n\n"
            f"HEALTH DATA:\n{health_summary}\n\n"
            f"Evaluate: \n"
            f"- HRV patterns (normal adult range 20-100ms, lower with age/disease)\n"
            f"- SpO2 levels (normal >95%, concerning <90%, critical <88%)\n"
            f"- Sleep architecture (normal deep sleep 15-25%, REM 20-25%)\n"
            f"- Vital signs relative to patient's baseline and conditions\n"
            f"- Activity levels appropriate for patient's fitness and health status"
        ),
        expected_output=(
            "A clinical assessment (2-3 sentences) identifying any significant anomalies or confirming "
            "stability, with specific reference to why findings are or are not concerning for this patient."
        ),
        agent=agent
    )

def create_enhanced_suggestion_task(date: datetime.date, health_summary: str, agent: Agent, personal_info: str) -> Task:
    """Create enhanced suggestion task with personalized recommendations."""
    return Task(
        description=(
            f"PATIENT PROFILE:\n{personal_info}\n\n"
            f"WELLNESS COACHING: Based on the health data for {date} and the patient's individual profile, "
            f"provide ONE specific, actionable health recommendation that is appropriate for their current "
            f"health status, medical conditions, and capabilities.\n\n"
            f"HEALTH DATA:\n{health_summary}\n\n"
            f"Consider:\n"
            f"- Patient's current fitness level and medical limitations\n"
            f"- Realistic and achievable goals\n"
            f"- Safety considerations for their health conditions\n"
            f"- Evidence-based lifestyle interventions\n"
            f"- Sleep hygiene, activity, stress management, or monitoring recommendations"
        ),
        expected_output=(
            "One specific, personalized health recommendation (1-2 sentences) that is medically appropriate "
            "and actionable for this patient's current health status and capabilities."
        ),
        agent=agent
    )

def create_enhanced_final_task(date: datetime.date, agent: Agent, context_tasks: List[Task], personal_info: str) -> Task:
    """Create enhanced final synthesis task."""
    return Task(
        description=(
            f"PATIENT PROFILE:\n{personal_info}\n\n"
            f"COMPREHENSIVE SYNTHESIS: Integrate the clinical interpretation, risk assessment, and "
            f"wellness recommendations to create a comprehensive daily health summary for {date}. "
            f"Provide a balanced, patient-centered report that acknowledges both positive findings "
            f"and areas for attention or improvement.\n\n"
            f"Create a summary that:\n"
            f"- Highlights key health insights from the day\n"
            f"- Addresses any significant concerns or positive trends\n"
            f"- Provides clear, actionable guidance\n"
            f"- Maintains an encouraging, supportive tone\n"
            f"- Considers the patient's overall health journey"
        ),
        expected_output=(
            "A comprehensive daily health summary (4-6 sentences) that integrates all analyses "
            "into clear, actionable insights for the patient and their healthcare team."
        ),
        agent=agent,
        context=context_tasks
    )

def create_enhanced_weekly_task(user_id: str, weekly_summaries: str, agent: Agent, personal_info: str) -> Task:
    """Create enhanced weekly trend analysis task."""
    return Task(
        description=(
            f"PATIENT PROFILE:\n{personal_info}\n\n"
            f"WEEKLY TREND ANALYSIS: Analyze the following 7 days of health summaries for patterns, "
            f"trends, and overall health trajectory. Identify improvements, concerns, and recommendations "
            f"for the upcoming week.\n\n"
            f"WEEKLY DATA:\n{weekly_summaries}\n\n"
            f"Focus on:\n"
            f"- Trends in sleep quality and duration\n"
            f"- Cardiovascular patterns (HRV trends)\n"
            f"- Respiratory stability (SpO2 patterns)\n"
            f"- Activity level consistency and progression\n"
            f"- Vital signs stability\n"
            f"- Overall health trajectory and areas for improvement"
        ),
        expected_output=(
            "A comprehensive weekly health report (5-7 sentences) identifying key trends, "
            "celebrating improvements, noting concerns, and providing specific recommendations "
            "for continued health optimization."
        ),
        agent=agent
    )

# ========================== MAIN PROCESSING FUNCTIONS ========================== #

def process_user_with_ai_agents(file_path: str, llm_instance) -> None:
    """Process user data with enhanced AI agent analysis."""
    try:
        filename = os.path.basename(file_path)
        user_id = filename.split('_')[1]
        
        print(f"🤖 Processing user file with AI agents: {filename}")
        
        # Load user data
        user_data = pd.read_excel(file_path, sheet_name=None)
        personal_info = user_data['Personal Info'].iloc[0]
        personal_info_summary = summarize_personal_info_enhanced(personal_info)
        
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
        
        daily_ai_summaries = []
        
        # Process each day
        for date in sorted(dates):
            print(f"  🧠 AI analysis for User {user_id} - Date: {date}")
            
            # Create comprehensive health summary
            health_summary = create_comprehensive_daily_summary(user_data, date)
            
            # Create tasks
            summary_task = create_enhanced_summary_task(date, health_summary, agents['summary'], personal_info_summary)
            anomaly_task = create_enhanced_anomaly_task(date, health_summary, agents['anomaly'], personal_info_summary)
            suggestion_task = create_enhanced_suggestion_task(date, health_summary, agents['suggestion'], personal_info_summary)
            final_task = create_enhanced_final_task(date, agents['final'], [summary_task, anomaly_task, suggestion_task], personal_info_summary)
            
            # Create crew and execute
            crew = Crew(
                agents=[agents['summary'], agents['anomaly'], agents['suggestion'], agents['final']],
                tasks=[summary_task, anomaly_task, suggestion_task, final_task],
                verbose=False
            )
            
            try:
                result = crew.kickoff()
                ai_summary = result.raw if hasattr(result, 'raw') else str(result)
                
                # Save daily AI analysis
                daily_filename = f"user_{user_id}_day_{date}_ai_analysis.txt"
                daily_path = os.path.join(DAILY_DIR, daily_filename)
                
                full_report = f"AI HEALTH ANALYSIS - {personal_info['Name']} - {date}\n"
                full_report += "=" * 60 + "\n\n"
                full_report += f"PATIENT PROFILE:\n{personal_info_summary}\n\n"
                full_report += f"RAW DATA SUMMARY:\n{health_summary}\n\n"
                full_report += f"AI CLINICAL ANALYSIS:\n{ai_summary}\n"
                
                with open(daily_path, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                
                daily_ai_summaries.append((date, ai_summary))
                
            except Exception as e:
                print(f"    ⚠️  Error in AI analysis for {date}: {str(e)}")
                continue
        
        # Generate weekly AI summary if enough days
        if len(daily_ai_summaries) >= 7:
            print(f"  📈 Generating weekly AI trend analysis for User {user_id}")
            
            # Prepare weekly data
            last_7_days = daily_ai_summaries[-7:]
            weekly_input = "\n\n".join([f"Date: {date}\nAnalysis: {summary}" for date, summary in last_7_days])
            
            # Create weekly task
            weekly_task = create_enhanced_weekly_task(user_id, weekly_input, agents['trend'], personal_info_summary)
            
            # Execute weekly analysis
            weekly_crew = Crew(
                agents=[agents['trend']],
                tasks=[weekly_task],
                verbose=False
            )
            
            try:
                weekly_result = weekly_crew.kickoff()
                weekly_ai_summary = weekly_result.raw if hasattr(weekly_result, 'raw') else str(weekly_result)
                
                # Save weekly AI analysis
                week_start = last_7_days[0][0]
                weekly_filename = f"user_{user_id}_week_{week_start}_ai_trends.txt"
                weekly_path = os.path.join(WEEKLY_DIR, weekly_filename)
                
                weekly_report = f"AI WEEKLY TREND ANALYSIS - {personal_info['Name']}\n"
                weekly_report += f"Week: {week_start} to {last_7_days[-1][0]}\n"
                weekly_report += "=" * 60 + "\n\n"
                weekly_report += f"PATIENT PROFILE:\n{personal_info_summary}\n\n"
                weekly_report += f"AI TREND ANALYSIS:\n{weekly_ai_summary}\n\n"
                weekly_report += "DAILY SUMMARIES:\n"
                for date, summary in last_7_days:
                    weekly_report += f"\n{date}:\n{summary}\n"
                
                with open(weekly_path, 'w', encoding='utf-8') as f:
                    f.write(weekly_report)
                
                print(f"    ✅ Weekly AI analysis saved: {weekly_filename}")
                
            except Exception as e:
                print(f"    ⚠️  Error in weekly AI analysis: {str(e)}")
        
        print(f"✅ Completed AI analysis for User {user_id}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""
    print("🤖 ENHANCED CREWAI HEALTH DATA ANALYZER")
    print("=" * 60)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🤖 AI reports directory: {OUTPUT_DIR}")
    print("=" * 60)
    
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
    print("🚀 Starting AI-powered health analysis...")
    
    # Process each file
    processed_count = 0
    for filename in sorted(excel_files):
        file_path = os.path.join(DATA_DIR, filename)
        process_user_with_ai_agents(file_path, llm)
        processed_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎉 AI ANALYSIS COMPLETE!")
    print(f"✅ Processed {processed_count} users with AI agents")
    
    # Summary statistics
    daily_files = len([f for f in os.listdir(DAILY_DIR) if f.endswith('.txt')])
    weekly_files = len([f for f in os.listdir(WEEKLY_DIR) if f.endswith('.txt')])
    
    print(f"\n📊 AI ANALYSIS RESULTS:")
    print(f"   • Daily AI analyses: {daily_files}")
    print(f"   • Weekly AI trend reports: {weekly_files}")
    print(f"   • Average daily reports per user: {daily_files/processed_count:.1f}")
    
    print(f"\n🧠 AI CAPABILITIES DEPLOYED:")
    print(f"   ✓ Clinical data interpretation")
    print(f"   ✓ Personalized risk assessment")
    print(f"   ✓ Evidence-based health recommendations")
    print(f"   ✓ Weekly trend analysis")
    print(f"   ✓ Comprehensive health synthesis")
    print(f"   ✓ Medical condition-aware insights")
    
    print("=" * 60)

if __name__ == "__main__":
    main()