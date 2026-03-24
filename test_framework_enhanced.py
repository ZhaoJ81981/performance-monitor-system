#!/usr/bin/env python3
"""
Enhanced Testing Framework for OpenClaw Projects
Comprehensive test suite covering:
1. Cron job monitoring
2. API endpoint testing
3. Skill functionality validation
4. Performance benchmarking
5. Integration testing
"""

import json
import subprocess
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenClawTestFramework:
    """Enhanced testing framework for OpenClaw projects"""
    
    def __init__(self, openclaw_path: str = None):
        self.openclaw_path = openclaw_path or "/Users/matrixx/.nvm/versions/node/v22.22.0/bin/openclaw"
        self.test_results = []
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], timeout: int = 60) -> Optional[Dict]:
        """Run command and return JSON result"""
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr[:200]}")
                return None
                
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout after {timeout}s")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def test_cron_jobs(self) -> Dict[str, Any]:
        """Test cron job functionality"""
        logger.info("Testing cron jobs...")
        
        # Get all cron jobs
        jobs = self.run_command([self.openclaw_path, 'cron', 'list', '--json', '--all'])
        if not jobs:
            return {"status": "failed", "message": "Failed to get cron jobs"}
        
        job_list = jobs.get('jobs', [])
        logger.info(f"Found {len(job_list)} cron jobs")
        
        # Test each job
        job_results = []
        for job in job_list[:5]:  # Test first 5 jobs
            job_id = job.get('id')
            job_name = job.get('name')
            
            # Get job runs
            runs = self.run_command([self.openclaw_path, 'cron', 'runs', '--id', job_id, '--limit', '3'])
            
            job_result = {
                "name": job_name,
                "id": job_id,
                "state": job.get('state', {}),
                "has_runs": bool(runs and runs.get('entries')),
                "run_count": len(runs.get('entries', [])) if runs else 0
            }
            job_results.append(job_result)
        
        return {
            "status": "success",
            "total_jobs": len(job_list),
            "tested_jobs": len(job_results),
            "job_details": job_results
        }
    
    def test_channels(self) -> Dict[str, Any]:
        """Test channel connectivity"""
        logger.info("Testing channels...")
        
        channels = self.run_command([self.openclaw_path, 'channels', 'list', '--json'])
        if not channels:
            return {"status": "failed", "message": "Failed to get channels"}
        
        channel_list = channels.get('chatChannels', [])
        logger.info(f"Found {len(channel_list)} channels")
        
        channel_results = []
        for channel in channel_list:
            channel_info = {
                "channel": channel.get('channel'),
                "accountId": channel.get('accountId'),
                "status": channel.get('status'),
                "enabled": channel.get('enabled')
            }
            channel_results.append(channel_info)
        
        return {
            "status": "success",
            "total_channels": len(channel_list),
            "channel_details": channel_results
        }
    
    def test_skills(self) -> Dict[str, Any]:
        """Test installed skills"""
        logger.info("Testing skills...")
        
        # Check skills directory
        skills_dir = Path.home() / ".agents" / "skills"
        if not skills_dir.exists():
            return {"status": "failed", "message": "Skills directory not found"}
        
        skills = []
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    skills.append({
                        "name": skill_dir.name,
                        "path": str(skill_dir),
                        "has_documentation": True
                    })
                else:
                    skills.append({
                        "name": skill_dir.name,
                        "path": str(skill_dir),
                        "has_documentation": False
                    })
        
        return {
            "status": "success",
            "total_skills": len(skills),
            "skills": skills[:10]  # Return first 10 skills
        }
    
    def test_wechat_integration(self) -> Dict[str, Any]:
        """Test WeChat integration specifically"""
        logger.info("Testing WeChat integration...")
        
        # Check if WeChat plugin is installed
        wechat_plugin_path = Path.home() / ".openclaw" / "extensions" / "openclaw-weixin"
        if not wechat_plugin_path.exists():
            return {
                "status": "warning",
                "message": "WeChat plugin not installed",
                "suggestion": "Run: npx -y @tencent-weixin/openclaw-weixin-cli install"
            }
        
        # Check WeChat skills
        wechat_skills = []
        skills_dir = Path.home() / ".agents" / "skills"
        for skill in skills_dir.glob("*wechat*"):
            if skill.is_dir():
                wechat_skills.append(skill.name)
        
        return {
            "status": "success",
            "plugin_installed": True,
            "plugin_path": str(wechat_plugin_path),
            "wechat_skills": wechat_skills,
            "skill_count": len(wechat_skills)
        }
    
    def test_performance(self) -> Dict[str, Any]:
        """Performance benchmarking"""
        logger.info("Running performance tests...")
        
        performance_results = []
        
        # Test command response time
        test_commands = [
            [self.openclaw_path, '--version'],
            [self.openclaw_path, 'cron', 'list', '--json'],
            [self.openclaw_path, 'channels', 'list', '--json']
        ]
        
        for cmd in test_commands:
            start = time.time()
            result = self.run_command(cmd, timeout=30)
            elapsed = time.time() - start
            
            performance_results.append({
                "command": ' '.join(cmd),
                "response_time": round(elapsed, 3),
                "success": result is not None
            })
        
        return {
            "status": "success",
            "performance_metrics": performance_results
        }
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        tests = [
            ("cron_jobs", self.test_cron_jobs),
            ("channels", self.test_channels),
            ("skills", self.test_skills),
            ("wechat_integration", self.test_wechat_integration),
            ("performance", self.test_performance)
        ]
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "test_duration": round(time.time() - self.start_time, 2),
            "tests": {}
        }
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                report["tests"][test_name] = result
                logger.info(f"Test '{test_name}': {result.get('status', 'unknown')}")
            except Exception as e:
                report["tests"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"Test '{test_name}' failed: {e}")
        
        # Calculate overall status
        test_statuses = [t.get("status") for t in report["tests"].values()]
        if all(status == "success" for status in test_statuses):
            report["overall_status"] = "success"
        elif any(status == "failed" for status in test_statuses):
            report["overall_status"] = "failed"
        else:
            report["overall_status"] = "warning"
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_dir: str = None):
        """Save test report to file"""
        if not output_dir:
            output_dir = Path.home() / ".openclaw" / "workspace" / "test_reports"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Also generate a markdown summary
        summary_file = output_path / f"test_summary_{timestamp}.md"
        self._generate_markdown_summary(report, summary_file)
        
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return str(report_file), str(summary_file)
    
    def _generate_markdown_summary(self, report: Dict[str, Any], output_path: Path):
        """Generate markdown summary of test results"""
        lines = [
            "# OpenClaw Test Report",
            f"**Generated at:** {report['timestamp']}",
            f"**Test duration:** {report['test_duration']} seconds",
            f"**Overall status:** **{report['overall_status'].upper()}**",
            "",
            "## Test Results",
            ""
        ]
        
        for test_name, test_result in report["tests"].items():
            status = test_result.get("status", "unknown")
            status_emoji = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
            
            lines.append(f"### {status_emoji} {test_name.replace('_', ' ').title()}")
            lines.append(f"- **Status:** {status}")
            
            if "message" in test_result:
                lines.append(f"- **Message:** {test_result['message']}")
            
            # Add specific details for each test type
            if test_name == "cron_jobs" and test_result.get("status") == "success":
                lines.append(f"- **Total jobs:** {test_result['total_jobs']}")
                lines.append(f"- **Tested jobs:** {test_result['tested_jobs']}")
                
            elif test_name == "channels" and test_result.get("status") == "success":
                lines.append(f"- **Total channels:** {test_result['total_channels']}")
                for channel in test_result['channel_details'][:3]:  # Show first 3
                    lines.append(f"  - {channel['channel']} ({channel['status']})")
                
            elif test_name == "skills" and test_result.get("status") == "success":
                lines.append(f"- **Total skills:** {test_result['total_skills']}")
                for skill in test_result['skills'][:5]:  # Show first 5
                    lines.append(f"  - {skill['name']}")
            
            elif test_name == "wechat_integration" and test_result.get("status") == "success":
                lines.append(f"- **Plugin installed:** {test_result['plugin_installed']}")
                lines.append(f"- **WeChat skills:** {test_result['skill_count']}")
                for skill in test_result.get('wechat_skills', [])[:3]:
                    lines.append(f"  - {skill}")
            
            elif test_name == "performance" and test_result.get("status") == "success":
                lines.append("- **Performance metrics:**")
                for metric in test_result['performance_metrics']:
                    lines.append(f"  - {metric['command']}: {metric['response_time']}s ({'✅' if metric['success'] else '❌'})")
            
            lines.append("")
        
        lines.append("## Recommendations")
        lines.append("")
        
        # Generate recommendations based on test results
        recommendations = []
        
        if report["tests"].get("cron_jobs", {}).get("status") == "failed":
            recommendations.append("Check OpenClaw cron service and ensure gateway is running")
        
        if report["tests"].get("channels", {}).get("status") == "failed":
            recommendations.append("Verify channel configurations and authentication")
        
        if report["tests"].get("wechat_integration", {}).get("status") == "warning":
            recommendations.append("Consider installing WeChat plugin for enhanced functionality")
        
        if not recommendations:
            recommendations.append("All systems operational. Consider adding more integration tests.")
        
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

def main():
    """Main entry point"""
    print("🚀 OpenClaw Enhanced Testing Framework")
    print("=" * 50)
    
    framework = OpenClawTestFramework()
    
    try:
        # Run all tests
        report = framework.generate_test_report()
        
        # Save reports
        json_report, md_report = framework.save_report(report)
        
        # Print summary
        print(f"\n📊 Test Report Generated:")
        print(f"  JSON Report: {json_report}")
        print(f"  Markdown Summary: {md_report}")
        print(f"\n📈 Overall Status: {report['overall_status'].upper()}")
        print(f"⏱️  Total Duration: {report['test_duration']} seconds")
        
        # Print test summary
        print("\n🧪 Test Results:")
        for test_name, test_result in report["tests"].items():
            status = test_result.get("status", "unknown")
            emoji = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
            print(f"  {emoji} {test_name}: {status}")
        
        return 0 if report["overall_status"] == "success" else 1
        
    except Exception as e:
        logger.error(f"Test framework error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())