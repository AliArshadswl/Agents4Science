#!/usr/bin/env python3
"""
main.py - Main execution script for Health AI Analysis System

This script provides multiple execution modes:
1. Standard multi-agent analysis
2. Single LLM comparison
3. Ablation studies
4. Comprehensive evaluation

Usage:
    python main.py --mode multi_agent --data_dir ./data
    python main.py --mode single_llm --data_dir ./data
    python main.py --mode ablation --data_dir ./data
    python main.py --mode compare_all --data_dir ./data

Author: Health AI Research Team
Date: 2024
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import Config
from src.utils.logger import setup_logging
from src.utils.file_manager import FileManager
from experiments.run_ablation import AblationRunner
from experiments.run_comparison import ComparisonRunner
from experiments.analyze_results import ResultsAnalyzer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Health AI Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run multi-agent analysis
    python main.py --mode multi_agent --data_dir ./realistic_health_data
    
    # Run single LLM comparison
    python main.py --mode single_llm --data_dir ./realistic_health_data
    
    # Run ablation studies
    python main.py --mode ablation --data_dir ./realistic_health_data
    
    # Run comprehensive comparison
    python main.py --mode compare_all --data_dir ./realistic_health_data
    
    # Analyze existing results
    python main.py --mode analyze --output_dir ./output
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multi_agent", "single_llm", "ablation", "compare_all", "analyze"],
        required=True,
        help="Execution mode"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./realistic_health_data",
        help="Directory containing health data Excel files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Maximum number of users to process (for testing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing where possible"
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)
    
    # Initialize configuration
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        max_users=args.max_users,
        parallel_processing=args.parallel
    )
    
    # Initialize file manager
    file_manager = FileManager(config)
    
    logger.info(f"🚀 Starting Health AI Analysis System")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Data Directory: {args.data_dir}")
    logger.info(f"   Output Directory: {args.output_dir}")
    logger.info(f"   Model: {args.model}")
    
    try:
        if args.mode == "multi_agent":
            from src.agents.health_agents import MultiAgentHealthAnalyzer
            analyzer = MultiAgentHealthAnalyzer(config)
            analyzer.run_analysis()
            
        elif args.mode == "single_llm":
            from src.agents.single_llm import SingleLLMHealthAnalyzer
            analyzer = SingleLLMHealthAnalyzer(config)
            analyzer.run_analysis()
            
        elif args.mode == "ablation":
            ablation_runner = AblationRunner(config)
            ablation_runner.run_all_ablations()
            
        elif args.mode == "compare_all":
            comparison_runner = ComparisonRunner(config)
            comparison_runner.run_comprehensive_comparison()
            
        elif args.mode == "analyze":
            analyzer = ResultsAnalyzer(config)
            analyzer.analyze_all_results()
            
        logger.info("✅ Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()