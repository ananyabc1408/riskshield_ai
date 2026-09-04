const $=id=>document.getElementById(id);
function money(n){return "₹"+Number(n).toLocaleString("en-IN",{maximumFractionDigits:0})}
async function api(path,opts){const r=await fetch(path,opts);if(!r.ok)throw new Error(await r.text());return r.json()}

async function loadDashboard(){
 const d=await api("/api/dashboard");
 $("kpis").innerHTML=[
  ["TRANSACTIONS",d.total_transactions.toLocaleString()],
  ["FRAUD LABELS",d.fraud_transactions.toLocaleString()],
  ["FRAUD RATE",d.fraud_rate+"%"],
  ["NETWORK",`${d.customers} customers · ${d.devices} devices`]
 ].map(x=>`<div class="kpi"><div class="l">${x[0]}</div><div class="v">${x[1]}</div></div>`).join("");
}

function loadDemo(){
 $("amount").value=25000;$("hour").value=23;$("failed_attempts").value=5;
 $("velocity_1h").value=8;$("velocity_24h").value=15;$("customer_avg_amount").value=1200;
 $("device_changed").checked=true;$("location_changed").checked=true;$("ring_score").value=.9;
}
$("scoreForm").addEventListener("submit",async e=>{
 e.preventDefault();
 const now=new Date(); now.setHours(Number($("hour").value));
 const p={
  transaction_id:$("transaction_id").value, timestamp:now.toISOString(),
  customer_id:$("customer_id").value, merchant_id:$("merchant_id").value,
  amount:Number($("amount").value),device_id:$("device_id").value,location:0,
  failed_attempts:Number($("failed_attempts").value),
  device_changed:$("device_changed").checked,location_changed:$("location_changed").checked,
  customer_avg_amount:Number($("customer_avg_amount").value),
  velocity_1h:Number($("velocity_1h").value),velocity_24h:Number($("velocity_24h").value),
  customer_frequency:Number($("customer_frequency").value),
  merchant_frequency:Number($("merchant_frequency").value),
  ring_score:Number($("ring_score").value)
 };
 $("result").innerHTML="<div class='empty'>Analysing behaviour + graph...</div>";
 try{
  const r=await api("/api/score",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
  const cls=r.risk_level.toLowerCase();
  $("result").innerHTML=`<div class="verdict">
   <div class="risk ${cls}">${r.risk_score}<small style="font-size:18px">/100</small></div>
   <div class="decision ${cls}">${r.risk_level} · ${r.decision}</div>
   <div class="bars">
    ${bar("Fraud probability",r.fraud_probability)}
    ${bar("Anomaly score",r.anomaly_score)}
    ${bar("Network/ring score",r.ring_score)}
   </div>
   <div class="reasons"><b>Why?</b>${r.reasons.map(x=>`<div class="reason">${x}</div>`).join("")}</div>
   ${r.counterfactual?`<div class="cf">⚡ ${r.counterfactual}</div>`:""}
  </div>`;
 }catch(err){$("result").innerHTML=`<div class="empty">Error: ${err.message}</div>`}
});
function bar(name,v){return `<div class="barrow"><span>${name}</span><div class="bar"><i style="width:${Math.round(v*100)}%"></i></div><b>${Math.round(v*100)}%</b></div>`}

async function loadMetrics(){
 const d=await api("/api/metrics"), a=d.training_validation,b=d.reference_test;
 $("metrics").innerHTML=`<div class="metric-grid">
 ${metric("Validation ROC-AUC",a.validation_auc)}
 ${metric("Validation PR-AUC",a.validation_pr_auc)}
 ${metric("Validation F1",a.validation_f1)}
 ${metric("Reference ensemble PR-AUC",b.ensemble_pr_auc)}
 ${metric("Reference rows",b.rows)}
 ${metric("Fraud labels",b.fraud_count)}
 </div>`;
}
function metric(k,v){return `<div class="metric"><span>${k}</span><b>${v==null?"—":Number(v).toFixed(3)}</b></div>`}

async function loadTransactions(){
 const rows=await api("/api/transactions?limit=10&risk_only=true");
 $("transactions").innerHTML=rows.map(r=>`<div class="tx"><div><div class="txid">${r.transaction_id}</div><div class="txsub">${r.customer_id} · ${money(r.amount)} · ${r.timestamp}</div></div><span class="badge">LABELED FRAUD</span></div>`).join("");
}
loadDashboard();loadMetrics();loadTransactions();
