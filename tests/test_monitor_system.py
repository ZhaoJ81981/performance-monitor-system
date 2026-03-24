#!/usr/bin/env python3
"""
Performance Monitor System Tests
Specialized tests for the Performance Monitor System project
"""

import json
import subprocess
import sys
import time
import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitorSystemTests:
    """Specialized tests for Performance Monitor System"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.test_results = []
        
    def test_project_structure(self):
        """Test if project has correct structure"""
        logger.info("Testing project structure...")
        
        required_dirs = [
            "src",
            "src/api",
            "src/collectors", 
            "src/ml_pipeline",
            "src/utils",
            "config",
            "data",
            "dashboards",
            "scripts"
        ]
        
        required_files = [
            "README.md",
            "docker-compose.yml",
            "requirements.txt",
            "Dockerfile.api",
            "Dockerfile.ml"
        ]
        
        results = {
            "directories": {},
            "files": {},
            "missing_dirs": [],
            "missing_files": []
        }
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            results["directories"][dir_path] = exists
            if not exists:
                results["missing_dirs"].append(dir_path)
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            results["files"][file_path] = exists
            if not exists:
                results["missing_files"].append(file_path)
        
        return {
            "status": "success" if not results["missing_dirs"] and not results["missing_files"] else "warning",
            "results": results
        }
    
    def test_docker_compose(self):
        """Test Docker Compose configuration"""
        logger.info("Testing Docker Compose...")
        
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            return {"status": "failed", "message": "docker-compose.yml not found"}
        
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
            
            # Basic validation
            has_services = "services:" in content
            has_api = "api:" in content or "web:" in content
            has_ml = "ml:" in content or "pipeline:" in content
            has_db = "db:" in content or "database:" in content or "postgres:" in content
            
            return {
                "status": "success",
                "has_services": has_services,
                "has_api": has_api,
                "has_ml": has_ml,
                "has_db": has_db,
                "file_valid": True
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_python_dependencies(self):
        """Test Python dependencies"""
        logger.info("Testing Python dependencies...")
        
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            return {"status": "warning", "message": "requirements.txt not found"}
        
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            dependencies = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    dependencies.append(line)
            
            # Check for essential dependencies
            essential_deps = ["flask", "fastapi", "pandas", "numpy", "scikit-learn"]
            found_essential = []
            missing_essential = []
            
            for dep in essential_deps:
                found = any(dep in d.lower() for d in dependencies)
                if found:
                    found_essential.append(dep)
                else:
                    missing_essential.append(dep)
            
            return {
                "status": "success" if not missing_essential else "warning",
                "total_dependencies": len(dependencies),
                "found_essential": found_essential,
                "missing_essential": missing_essential,
                "dependencies": dependencies[:10]  # First 10
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_ml_pipeline(self):
        """Test ML pipeline components"""
        logger.info("Testing ML pipeline...")
        
        ml_dir = self.project_root / "src" / "ml_pipeline"
        if not ml_dir.exists():
            return {"status": "warning", "message": "ML pipeline directory not found"}
        
        ml_files = list(ml_dir.glob("*.py"))
        
        # Check for key ML components
        components = {
            "data_processor": any("process" in f.name.lower() or "preprocess" in f.name.lower() for f in ml_files),
            "model_trainer": any("train" in f.name.lower() or "model" in f.name.lower() for f in ml_files),
            "predictor": any("predict" in f.name.lower() or "inference" in f.name.lower() for f in ml_files),
            "evaluator": any("eval" in f.name.lower() or "score" in f.name.lower() for f in ml_files)
        }
        
        return {
            "status": "success",
            "ml_files_count": len(ml_files),
            "ml_files": [f.name for f in ml_files[:5]],
            "components": components,
            "has_all_components": all(components.values())
        }
    
    def test_api_endpoints(self):
        """Test API structure"""
        logger.info("Testing API endpoints...")
        
        api_dir = self.project_root / "src" / "api"
        if not api_dir.exists():
            return {"status": "warning", "message": "API directory not found"}
        
        api_files = list(api_dir.glob("*.py"))
        
        # Check for common API patterns
        endpoints_present = {
            "health_check": any("health" in f.name.lower() for f in api_files),
            "metrics": any("metric" in f.name.lower() for f in api_files),
            "predict": any("predict" in f.name.lower() or "infer" in f.name.lower() for f in api_files),
            "collect": any("collect" in f.name.lower() or "data" in f.name.lower() for f in api_files)
        }
        
        return {
            "status": "success",
            "api_files_count": len(api_files),
            "api_files": [f.name for f in api_files[:5]],
            "endpoints": endpoints_present
        }
    
    def test_collectors(self):
        """Test data collectors"""
        logger.info("Testing data collectors...")
        
        collectors_dir = self.project_root / "src" / "collectors"
        if not collectors_dir.exists():
            return {"status": "warning", "message": "Collectors directory not found"}
        
        collector_files = list(collectors_dir.glob("*.py"))
        
        # Check for different collector types
        collector_types = {
            "system_metrics": any("system" in f.name.lower() or "cpu" in f.name.lower() or "memory" in f.name.lower() for f in collector_files),
            "application_metrics": any("app" in f.name.lower() or "service" in f.name.lower() for f in collector_files),
            "business_metrics": any("business" in f.name.lower() or "kpi" in f.name.lower() for f in collector_files),
            "external_metrics": any("external" in f.name.lower() or "api" in f.name.lower() for f in collector_files)
        }
        
        return {
            "status": "success",
            "collector_files_count": len(collector_files),
            "collector_files": [f.name for f in collector_files[:5]],
            "collector_types": collector_types
        }
    
    def test_scripts(self):
        """Test utility scripts"""
        logger.info("Testing utility scripts...")
        
        scripts_dir = self.project_root / "scripts"
        if not scripts_dir.exists():
            return {"status": "warning", "message": "Scripts directory not found"}
        
        script_files = list(scripts_dir.glob("*"))
        
        # Check script types
        script_types = {
            "bash_scripts": any(f.suffix == '.sh' for f in script_files),
            "python_scripts": any(f.suffix == '.py' for f in script_files),
            "deployment": any("deploy" in f.name.lower() or "setup" in f.name.lower() for f in script_files),
            "maintenance": any("backup" in f.name.lower() or "clean" in f.name.lower() or "migrate" in f.name.lower() for f in script_files)
        }
        
        return {
            "status": "success",
            "script_files_count": len(script_files),
            "script_files": [f.name for f in script_files[:5]],
            "script_types": script_types
        }
    
    def run_all_tests(self):
        """Run all tests for Monitor System"""
        logger.info("Running all Performance Monitor System tests...")
        
        tests = [
            ("project_structure", self.test_project_structure),
            ("docker_compose", self.test_docker_compose),
            ("python_dependencies", self.test_python_dependencies),
            ("ml_pipeline", self.test_ml_pipeline),
            ("api_endpoints", self.test_api_endpoints),
            ("collectors", self.test_collectors),
            ("scripts", self.test_scripts)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                logger.info(f"Test '{test_name}': {result.get('status', 'unknown')}")
            except Exception as e:
                results[test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"Test '{test_name}' failed: {e}")
        
        # Calculate overall status
        test_statuses = [r.get("status") for r in results.values()]
        if all(status == "success" for status in test_statuses):
            overall_status = "success"
        elif any(status == "failed" for status in test_statuses):
            overall_status = "failed"
        else:
            overall_status = "warning"
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "project": "Performance Monitor System",
            "project_root": str(self.project_root),
            "overall_status": overall_status,
            "tests": results
        }
    
    def generate_report(self, results):
        """Generate test report"""
        report_dir = self.project_root / "test_reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"monitor_test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Generate markdown summary
        summary_file = report_dir / f"monitor_test_summary_{timestamp}.md"
        self._generate_markdown_summary(results, summary_file)
        
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return str(report_file), str(summary_file)
    
    def _generate_markdown_summary(self, results, output_path):
        """Generate markdown summary"""
        lines = [
            "# Performance Monitor System Test Report",
            f"**Generated at:** {results['timestamp']}",
            f"**Project:** {results['project']}",
            f"**Overall status:** **{results['overall_status'].upper()}**",
            "",
            "## Test Results",
            ""
        ]
        
        for test_name, test_result in results["tests"].items():
            status = test_result.get("status", "unknown")
            status_emoji = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
            
            lines.append(f"### {status_emoji} {test_name.replace('_', ' ').title()}")
            lines.append(f"- **Status:** {status}")
            
            # Add specific details for each test
            if test_name == "project_structure" and "results" in test_result:
                res = test_result["results"]
                lines.append(f"- **Missing directories:** {len(res['missing_dirs'])}")
                lines.append(f"- **Missing files:** {len(res['missing_files'])}")
                
            elif test_name == "python_dependencies" and "total_dependencies" in test_result:
                lines.append(f"- **Total dependencies:** {test_result['total_dependencies']}")
                if test_result.get('missing_essential'):
                    lines.append(f"- **Missing essential:** {', '.join(test_result['missing_essential'])}")
                    
            elif test_name == "ml_pipeline" and "ml_files_count" in test_result:
                lines.append(f"- **ML files:** {test_result['ml_files_count']}")
                lines.append(f"- **Has all components:** {test_result['has_all_components']}")
                
            elif test_name == "api_endpoints" and "api_files_count" in test_result:
                lines.append(f"- **API files:** {test_result['api_files_count']}")
                
            elif test_name == "collectors" and "collector_files_count" in test_result:
                lines.append(f"- **Collector files:** {test_result['collector_files_count']}")
                
            elif test_name == "scripts" and "script_files_count" in test_result:
                lines.append(f"- **Script files:** {test_result['script_files_count']}")
            
            lines.append("")
        
        lines.append("## Recommendations")
        lines.append("")
        
        recommendations = []
        
        # Generate recommendations based on test results
        if results["tests"].get("project_structure", {}).get("status") == "warning":
            missing = results["tests"]["project_structure"]["results"]
            if missing["missing_dirs"]:
                recommendations.append(f"Create missing directories: {', '.join(missing['missing_dirs'][:3])}")
            if missing["missing_files"]:
                recommendations.append(f"Create missing files: {', '.join(missing['missing_files'][:3])}")
        
        if results["tests"].get("python_dependencies", {}).get("status") == "warning":
            missing = results["tests"]["python_dependencies"].get("missing_essential", [])
            if missing:
                recommendations.append(f"Add essential dependencies: {', '.join(missing)}")
        
        if results["tests"].get("ml_pipeline", {}).get("has_all_components") == False:
            recommendations.append("Consider adding missing ML pipeline components")
        
        if not recommendations:
            recommendations.append("Project structure looks good! Consider adding integration tests.")
        
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

def main():
    """Main entry point"""
    print("🔧 Performance Monitor System Test Suite")
    print("=" * 50)
    
    project_root = Path.home() / ".openclaw" / "workspace" / "performance-monitor-system"
    
    tester = MonitorSystemTests(project_root)
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Generate reports
        json_report, md_report = tester.generate_report(results)
        
        # Print summary
        print(f"\n📊 Test Report Generated:")
        print(f"  JSON Report: {json_report}")
        print(f"  Markdown Summary: {md_report}")
        print(f"\n📈 Overall Status: {results['overall_status'].upper()}")
        
        # Print test summary
        print("\n🧪 Test Results:")
        for test_name, test_result in results["tests"].items():
            status = test_result.get("status", "unknown")
            emoji = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
            print(f"  {emoji} {test_name}: {status}")
        
        return 0 if results["overall_status"] == "success" else 1
        
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())