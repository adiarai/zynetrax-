from intelligence.trust_engine import TrustEngine
from intelligence.dependency_detector import DependencyDetector
from intelligence.visualization_engine import VisualizationEngine
from intelligence.executive_report_engine import ExecutiveReportEngine
from intelligence.structural_intelligence_engine import StructuralIntelligenceEngine
from intelligence.driver_modeling_engine import DriverModelingEngine
from intelligence.scenario_simulation_engine import ScenarioSimulationEngine
from intelligence.executive_recommendation_engine import ExecutiveRecommendationEngine
from intelligence.pattern_intelligence_engine import PatternIntelligenceEngine
from intelligence.dataset_behavior_engine import DatasetBehaviorEngine
from intelligence.outcome_discovery_engine import OutcomeDiscoveryEngine
from intelligence.temporal_intelligence_engine import TemporalIntelligenceEngine
from intelligence.causal_intelligence_engine import CausalIntelligenceEngine
from intelligence.meta_learning_engine import MetaLearningEngine
from intelligence.semantic_embedding_engine import SemanticEmbeddingEngine
from intelligence.autonomous_feature_engineering_engine import AutonomousFeatureEngineeringEngine

# NEW ENGINES
from intelligence.business_context_engine import BusinessContextEngine
from intelligence.driver_filtering_engine import DriverFilteringEngine
from intelligence.knowledge_memory_engine import KnowledgeMemoryEngine
from intelligence.dataset_benchmark_engine import DatasetBenchmarkEngine
from intelligence.autonomous_hypothesis_engine import AutonomousHypothesisEngine
from intelligence.natural_language_query_engine import NaturalLanguageQueryEngine
from intelligence.autonomous_experimentation_engine import AutonomousExperimentationEngine
from intelligence.global_strategy_optimizer_engine import GlobalStrategyOptimizerEngine
from intelligence.universal_dataset_intelligence_engine import UniversalDatasetIntelligenceEngine

import argparse
import os
import json

from intelligence.semantic_detector import SemanticColumnDetector
from intelligence.insight_generator import InsightGenerator
from intelligence.relationship_detector import RelationshipDetector

from core.loader import FileLoader
from core.normalizer import AdvancedDataNormalizer
from core.validator import NumericValidator
from core.schema import SmartSchemaDetector
from core.roles import AdvancedColumnRoleDetector
from core.quality import DataQualityAnalyzer
from core.features import FeatureEngineer
from core.exporter import ExportManager

from text_engine.text_processor import TextProcessor


def run_pipeline(file_paths, output_prefix="output"):

    print("🚀 Starting AI Data Engine...\n")

    tables = {}
    reports = {}

    for path in file_paths:

        table_name = os.path.splitext(os.path.basename(path))[0]

        print(f"\n📂 Processing table: {table_name}")

        extension = os.path.splitext(path)[1].lower()
        text_meta = None

        if extension == ".txt":
            print("📝 Processing unstructured text...")
            df, text_meta = TextProcessor.process_text_file(path)
        else:
            df = FileLoader.load_file(path)

        # BUSINESS CONTEXT

        context_report = BusinessContextEngine.detect(df)

        print("\n🏢 Business Context Detected:")
        print(context_report)

        past_knowledge = KnowledgeMemoryEngine.recall(
            context_report["detected_context"]
        )

        print("\n📚 Past Knowledge Patterns:")
        print(past_knowledge)

        # CORE PIPELINE

        df = AdvancedDataNormalizer.normalize(df)
        df, numeric_issues = NumericValidator.validate(df)

        schema = SmartSchemaDetector.analyze_dataframe(df)

        semantic_map = SemanticColumnDetector.detect(df)

        embedding_semantics = SemanticEmbeddingEngine.analyze(df)

        print("\n🧠 Semantic Embedding Mapping:")
        print(embedding_semantics)

        # UNIVERSAL DATASET INTELLIGENCE

        universal_semantics = UniversalDatasetIntelligenceEngine.analyze(df)

        print("\n🌍 Universal Dataset Intelligence:")
        print(universal_semantics)

        roles = AdvancedColumnRoleDetector.detect_roles(df)

        quality = DataQualityAnalyzer.analyze(df)

        insights = InsightGenerator.generate(df, schema, quality, semantic_map)

        df = FeatureEngineer.engineer(df)

        # NEW AUTONOMOUS FEATURE ENGINEERING

        df, auto_features = AutonomousFeatureEngineeringEngine.generate(df)

        print("\n🧠 Autonomous Feature Engineering:")
        print(auto_features)

        # PATTERN INTELLIGENCE

        pattern_report = PatternIntelligenceEngine.analyze(df)

        print("\n🔎 Pattern Intelligence Summary:")
        print(pattern_report)

        behavior_report = DatasetBehaviorEngine.analyze(df)

        print("\n📊 Dataset Behavior Summary:")
        print(behavior_report)

        temporal_report = TemporalIntelligenceEngine.analyze(df)

        print("\n⏳ Temporal Intelligence Summary:")
        print(temporal_report)

        outcome_report = OutcomeDiscoveryEngine.analyze(df)

        print("\n🎯 Outcome Discovery Summary:")
        print(outcome_report)

        # STRUCTURAL INTELLIGENCE

        structural_report = StructuralIntelligenceEngine.analyze(df)

        print("\n🧠 Structural Intelligence Summary:")
        print(structural_report)

        # DRIVER FILTERING

        driver_filter = DriverFilteringEngine.filter(df, structural_report)

        print("\n🧹 Driver Filtering:")
        print(driver_filter)

        filtered_drivers = driver_filter["filtered_drivers"]

        # DRIVER MODELING

        driver_model_results = DriverModelingEngine.analyze(
            df,
            structural_report,
            filtered_drivers
        )

        print("\n🤖 Driver Modeling Summary:")
        print(driver_model_results)

        # BENCHMARKING

        benchmark_comparison = DatasetBenchmarkEngine.compare(
            context_report,
            driver_model_results
        )

        print("\n📊 Benchmark Comparison:")
        print(benchmark_comparison)

        benchmark_store = DatasetBenchmarkEngine.store(
            context_report,
            driver_model_results
        )

        print("\n📊 Benchmark Storage Update:")
        print(benchmark_store)

        # CAUSAL INTELLIGENCE

        causal_report = CausalIntelligenceEngine.analyze(df, structural_report)

        print("\n🧠 Causal Intelligence Summary:")
        print(causal_report)

        memory_report = MetaLearningEngine.learn(
            structural_report,
            causal_report
        )

        print("\n🧠 Meta Learning Memory:")
        print(memory_report)

        knowledge_update = KnowledgeMemoryEngine.learn(
            structural_report,
            driver_model_results,
            context_report
        )

        print("\n🧠 Knowledge Memory Update:")
        print(knowledge_update)

        # SCENARIO SIMULATION

        scenario_results = ScenarioSimulationEngine.simulate(
            df,
            structural_report
        )

        print("\n📊 Scenario Simulation Summary:")
        print(scenario_results)

        # AUTONOMOUS EXPERIMENTATION

        experiment_results = AutonomousExperimentationEngine.run(
            df,
            structural_report,
            driver_model_results
        )

        print("\n🧪 Autonomous Strategy Experiments:")
        print(experiment_results)

        # GLOBAL OPTIMIZER

        optimizer_results = GlobalStrategyOptimizerEngine.optimize(
            df,
            structural_report,
            driver_model_results
        )

        print("\n🚀 Global Strategy Optimizer:")
        print(optimizer_results)

        # AUTONOMOUS HYPOTHESIS

        hypothesis_report = AutonomousHypothesisEngine.generate(
            structural_report,
            driver_model_results,
            scenario_results
        )

        print("\n🧠 Autonomous Hypotheses:")
        print(hypothesis_report)

        # EXECUTIVE RECOMMENDATIONS

        recommendation_report = ExecutiveRecommendationEngine.generate(
            structural_report,
            driver_model_results,
            scenario_results
        )

        print("\n🧠 Executive Intelligence Summary:")
        print(recommendation_report)

        dependencies = DependencyDetector.detect(df)

        tables[table_name] = df

        trust_scores = TrustEngine.evaluate(
            schema=schema,
            quality=quality,
            relationships=None,
            text_meta=text_meta
        )

        VisualizationEngine.generate_all(
            df,
            dependencies,
            trust_scores,
            table_name,
            base_output=output_prefix
        )

        reports[table_name] = {
            "universal_semantics": universal_semantics,
            "auto_features": auto_features,
            "structural_intelligence": structural_report,
            "driver_modeling": driver_model_results,
            "scenario_simulation": scenario_results,
            "autonomous_experiments": experiment_results,
            "global_optimizer": optimizer_results,
            "executive_recommendations": recommendation_report,
            "autonomous_hypotheses": hypothesis_report
        }

    print("\n💬 AI Analyst Ready. Ask questions about the dataset.")
    print("Type 'exit' to quit.\n")

    while True:

        user_query = input("Ask AI Analyst: ")

        if user_query.lower() == "exit":
            break

        for table_name in reports:

            response = NaturalLanguageQueryEngine.answer(
                user_query,
                reports[table_name]
            )

            print("\nAI Answer:")
            print(response)

    print("\n✅ Pipeline Complete!")

    return tables, reports


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AI Data Engine CLI")

    parser.add_argument("files", nargs="+", help="Paths to input data files")

    parser.add_argument("--output", default="output", help="Output file prefix")

    args = parser.parse_args()

    run_pipeline(args.files, args.output)