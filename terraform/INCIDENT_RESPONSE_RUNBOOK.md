# 🚨 Multi-Agent System Incident Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to incidents in the Cabruca Segmentation Multi-Agent System.

---

## 📋 Quick Reference

### Critical Resources
- **Region**: sa-east-1 (São Paulo)
- **CloudWatch Dashboard**: [Agent Monitoring Dashboard](https://console.aws.amazon.com/cloudwatch/home?region=sa-east-1#dashboards:name=cabruca-segmentation-prod-agents-dashboard)
- **AgentOps Dashboard**: [AgentOps Monitoring](https://app.agentops.ai)
- **Primary Contact**: DevOps Team
- **Escalation**: Engineering Lead → CTO

### Severity Levels
- **P1 (Critical)**: Complete system outage, data loss risk
- **P2 (High)**: Major feature unavailable, significant performance degradation
- **P3 (Medium)**: Minor feature issues, isolated agent failures
- **P4 (Low)**: Non-critical issues, monitoring alerts

---

## 🔥 Common Incidents and Resolutions

### 1. Agent Lambda Function Errors

#### Symptoms
- High error rate in CloudWatch metrics
- AgentOps showing failed sessions
- API returning 500 errors

#### Diagnosis
```bash
# Check recent logs
aws logs tail /aws/lambda/cabruca-segmentation-prod-${AGENT_NAME}-agent --follow

# Check error metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=cabruca-segmentation-prod-${AGENT_NAME}-agent \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Sum
```

#### Resolution Steps
1. **Identify the failing agent**:
   ```bash
   ./test_agents.sh  # Run test suite to identify failures
   ```

2. **Check Lambda configuration**:
   ```bash
   aws lambda get-function-configuration \
     --function-name cabruca-segmentation-prod-${AGENT_NAME}-agent
   ```

3. **Common fixes**:
   - **Timeout issues**: Increase timeout in Terraform and redeploy
   - **Memory issues**: Increase memory allocation
   - **Permission issues**: Check IAM role policies
   
4. **Quick rollback**:
   ```bash
   # Rollback to previous version
   aws lambda update-function-code \
     --function-name cabruca-segmentation-prod-${AGENT_NAME}-agent \
     --s3-bucket cabruca-segmentation-prod-agent-artifacts-${ACCOUNT_ID} \
     --s3-key backups/lambda/${AGENT_NAME}/previous.zip
   ```

---

### 2. DynamoDB Throttling

#### Symptoms
- Throttled requests in CloudWatch
- Slow agent responses
- Tasks stuck in "pending" state

#### Diagnosis
```bash
# Check DynamoDB metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name UserErrors \
  --dimensions Name=TableName,Value=cabruca-segmentation-prod-agent-state \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Sum
```

#### Resolution Steps
1. **Immediate mitigation**:
   ```bash
   # Switch to on-demand billing (if using provisioned)
   aws dynamodb update-table \
     --table-name cabruca-segmentation-prod-agent-${TABLE} \
     --billing-mode PAY_PER_REQUEST
   ```

2. **Clear stuck tasks**:
   ```python
   # Python script to clear old tasks
   import boto3
   from datetime import datetime, timedelta
   
   dynamodb = boto3.resource('dynamodb')
   table = dynamodb.Table('cabruca-segmentation-prod-agent-tasks')
   
   # Scan for old pending tasks
   response = table.scan(
       FilterExpression='status = :status AND timestamp < :old_time',
       ExpressionAttributeValues={
           ':status': 'pending',
           ':old_time': int((datetime.now() - timedelta(hours=1)).timestamp())
       }
   )
   
   # Delete old tasks
   for item in response['Items']:
       table.delete_item(Key={'task_id': item['task_id']})
   ```

---

### 3. S3 Access Issues

#### Symptoms
- Agents unable to read/write artifacts
- "Access Denied" errors in logs
- Missing processed outputs

#### Diagnosis
```bash
# Check bucket policy
aws s3api get-bucket-policy --bucket ${BUCKET_NAME}

# Check bucket versioning
aws s3api get-bucket-versioning --bucket ${BUCKET_NAME}

# List recent objects
aws s3 ls s3://${BUCKET_NAME}/ --recursive --summarize | tail -20
```

#### Resolution Steps
1. **Verify IAM permissions**:
   ```bash
   # Check Lambda execution role
   aws iam get-role-policy \
     --role-name cabruca-segmentation-prod-${AGENT}-agent-role \
     --policy-name cabruca-segmentation-prod-${AGENT}-agent-policy
   ```

2. **Fix bucket permissions**:
   ```bash
   # Update bucket policy
   aws s3api put-bucket-policy --bucket ${BUCKET_NAME} --policy file://bucket-policy.json
   ```

---

### 4. High Lambda Costs / Runaway Executions

#### Symptoms
- Unexpected AWS bill increase
- Continuous Lambda invocations
- CloudWatch showing high invocation count

#### Diagnosis
```bash
# Check invocation metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=cabruca-segmentation-prod-${AGENT_NAME}-agent \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600 \
  --statistics Sum
```

#### Resolution Steps
1. **Immediate stop** (Emergency):
   ```bash
   # Disable all triggers
   for agent in manager engineer qa researcher data_processor; do
     aws lambda put-function-concurrency \
       --function-name cabruca-segmentation-prod-${agent}-agent \
       --reserved-concurrent-executions 0
   done
   ```

2. **Investigate cause**:
   ```bash
   # Check EventBridge rules
   aws events list-rules --name-prefix cabruca-segmentation
   
   # Check S3 event notifications
   aws s3api get-bucket-notification-configuration \
     --bucket cabruca-segmentation-prod-agent-queue
   ```

3. **Apply cost controls**:
   ```bash
   # Set conservative concurrency limits
   aws lambda put-function-concurrency \
     --function-name cabruca-segmentation-prod-${agent}-agent \
     --reserved-concurrent-executions 5
   ```

---

### 5. AgentOps Connection Issues

#### Symptoms
- No data in AgentOps dashboard
- "Failed to initialize AgentOps" in logs
- Missing session tracking

#### Resolution Steps
1. **Verify API key**:
   ```bash
   # Check environment variable
   aws lambda get-function-configuration \
     --function-name cabruca-segmentation-prod-${AGENT}-agent \
     --query 'Environment.Variables.AGENTOPS_API_KEY'
   ```

2. **Update API key**:
   ```bash
   # Update environment variable
   aws lambda update-function-configuration \
     --function-name cabruca-segmentation-prod-${AGENT}-agent \
     --environment Variables={AGENTOPS_API_KEY=your-new-key}
   ```

---

## 📊 Monitoring and Alerting

### Key Metrics to Watch

1. **Lambda Metrics**:
   - Error rate > 1%
   - Duration > 80% of timeout
   - Throttles > 0
   - Concurrent executions > 80% of limit

2. **DynamoDB Metrics**:
   - ConsumedReadCapacityUnits
   - ConsumedWriteCapacityUnits
   - UserErrors > 0
   - SystemErrors > 0

3. **S3 Metrics**:
   - 4xx errors > 1%
   - 5xx errors > 0
   - Request latency > 1000ms

### Alert Response Times

| Severity | Response Time | Escalation |
|----------|--------------|------------|
| P1 | 15 minutes | Immediate |
| P2 | 30 minutes | 1 hour |
| P3 | 2 hours | 4 hours |
| P4 | 24 hours | 48 hours |

---

## 🛠 Maintenance Procedures

### Daily Health Checks
```bash
# Run the test suite
./test_agents.sh

# Check cost trends
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

### Weekly Tasks
1. Review CloudWatch logs for errors
2. Check DynamoDB table sizes and costs
3. Clean up old S3 artifacts
4. Update AgentOps dashboard configurations
5. Review and adjust alarm thresholds

### Monthly Tasks
1. Cost analysis and optimization
2. Security audit (IAM policies, bucket policies)
3. Performance baseline review
4. Disaster recovery test

---

## 🔄 Disaster Recovery

### Backup Procedures
```bash
# Backup DynamoDB tables
for table in agent-state agent-memory agent-tasks; do
  aws dynamodb create-backup \
    --table-name cabruca-segmentation-prod-${table} \
    --backup-name ${table}-$(date +%Y%m%d-%H%M%S)
done

# Backup Lambda functions
for agent in manager engineer qa researcher data_processor; do
  aws lambda get-function \
    --function-name cabruca-segmentation-prod-${agent}-agent \
    --query 'Code.Location' \
    --output text | xargs wget -O backups/${agent}-$(date +%Y%m%d).zip
done
```

### Recovery Procedures
```bash
# Restore DynamoDB from backup
aws dynamodb restore-table-from-backup \
  --target-table-name cabruca-segmentation-prod-${TABLE}-restored \
  --backup-arn ${BACKUP_ARN}

# Restore Lambda from backup
aws lambda update-function-code \
  --function-name cabruca-segmentation-prod-${AGENT}-agent \
  --zip-file fileb://backups/${AGENT}-${DATE}.zip
```

---

## 📞 Contact Information

### Primary Contacts
- **DevOps Lead**: +55 11 9XXXX-XXXX
- **Engineering Lead**: +55 11 9XXXX-XXXX
- **AWS Support**: [AWS Support Console](https://console.aws.amazon.com/support)

### Escalation Path
1. On-call Engineer (PagerDuty)
2. DevOps Lead
3. Engineering Lead
4. CTO

### External Dependencies
- **AgentOps Support**: support@agentops.ai
- **AWS Support Plan**: Business Support
- **Monitoring Tools**: CloudWatch, AgentOps, PagerDuty

---

## 📝 Post-Incident Review

After resolving any P1 or P2 incident:

1. **Document the incident**:
   - Timeline of events
   - Root cause analysis
   - Actions taken
   - Impact assessment

2. **Update runbook**:
   - Add new scenarios discovered
   - Update resolution steps
   - Improve monitoring

3. **Implement preventive measures**:
   - Add new alarms
   - Update auto-scaling policies
   - Improve error handling

4. **Share learnings**:
   - Team retrospective
   - Update documentation
   - Training if needed

---

## 🔧 Useful Commands Reference

```bash
# Get all agent statuses
for agent in manager engineer qa researcher data_processor; do
  echo "=== $agent ==="
  aws lambda get-function \
    --function-name cabruca-segmentation-prod-${agent}-agent \
    --query 'Configuration.State' \
    --output text
done

# Check recent errors across all agents
for agent in manager engineer qa researcher data_processor; do
  echo "=== $agent errors ==="
  aws logs filter-log-events \
    --log-group-name /aws/lambda/cabruca-segmentation-prod-${agent}-agent \
    --filter-pattern ERROR \
    --start-time $(($(date +%s) - 3600))000 \
    --query 'events[*].message' \
    --output text | head -5
done

# Emergency shutdown all agents
terraform apply -var="enable_agents=false" -auto-approve

# Emergency scale down
aws application-autoscaling put-scaling-policy \
  --policy-name emergency-scale-down \
  --service-namespace lambda \
  --resource-id function:cabruca-segmentation-prod-*-agent \
  --scalable-dimension lambda:function:ProvisionedConcurrencyCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scale-down-policy.json
```

---

**Last Updated**: March 2024
**Version**: 1.0.0
**Review Schedule**: Monthly